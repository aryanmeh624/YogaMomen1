package com.example.yogamomen

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.view.Surface
import android.widget.Toast
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    private var previousKeypoints: List<Pair<Float, Float>>? = null
    private val SMOOTHING_FACTOR = 0.7f  // Higher value increases smoothing
    private lateinit var tflite: Interpreter
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView

    companion object {
        private const val REQUEST_CODE_CAMERA_PERMISSION = 10
        private const val MODEL_WIDTH = 513
        private const val MODEL_HEIGHT = 257
        private const val NUM_KEYPOINTS = 17
        private const val CONFIDENCE_THRESHOLD = 0.4f  // Lowered confidence threshold
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Set the content view first
        setContentView(R.layout.activity_main)

        // Initialize views after setting the content view
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)

        // Check if overlayView and previewView are initialized
        if (overlayView == null || previewView == null) {
            Log.e("MainActivity", "Failed to initialize previewView or overlayView")
            return
        }

        overlayView.post {
            if (overlayView.width > 0 && overlayView.height > 0) {
                Log.d("MainActivity", "OverlayView dimensions are initialized: ${overlayView.width} x ${overlayView.height}")
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    startCamera()
                }
            }
        }

        // Initialize the interpreter
        initializeInterpreter()

        // Check for camera permissions
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQUEST_CODE_CAMERA_PERMISSION)
        } else {
            startCamera()  // Start camera only after views are initialized
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_CAMERA_PERMISSION && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission is required to use this feature", Toast.LENGTH_SHORT).show()
        }
    }

    private fun initializeInterpreter() {
        val options = Interpreter.Options()
        tflite = Interpreter(loadModelFile(), options)

        // Log the output tensors
        val outputCount = tflite.outputTensorCount
        Log.d("Interpreter", "Number of outputs: $outputCount")
        for (i in 0 until outputCount) {
            val tensor = tflite.getOutputTensor(i)
            Log.d("Interpreter", "Output tensor $i: name=${tensor.name()}, shape=${tensor.shape().contentToString()}, dataType=${tensor.dataType()}")
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = assets.openFd("posenet_mobilenet.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(previewView.display.rotation)
                .build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(previewView.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalyzer.setAnalyzer(ContextCompat.getMainExecutor(this)) { imageProxy ->
                processImage(imageProxy)
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Log.e("CameraX", "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(imageProxy: ImageProxy) {
        val bitmap = imageProxyToBitmap(imageProxy)
        if (bitmap != null) {
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, MODEL_WIDTH, MODEL_HEIGHT, true)
            val inputBuffer = convertBitmapToByteBuffer(scaledBitmap)

            // Get output tensor shape
            val heatmapsShape = tflite.getOutputTensor(0).shape()
            val heatmapBatchSize = heatmapsShape[0]
            val hmap_h = heatmapsShape[1]
            val heatmapWidth = heatmapsShape[2]
            val numKeypoints = heatmapsShape[3]

            // Prepare output array
            val heatmaps = Array(heatmapBatchSize) { Array(hmap_h) { Array(heatmapWidth) { FloatArray(numKeypoints) } } }

            val outputMap = mutableMapOf<Int, Any>(0 to heatmaps)
            tflite.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

            val keypoints = extractKeypointsFromHeatmap(heatmaps, overlayView.width.toFloat(), overlayView.height.toFloat())

            drawKeypoints(keypoints)
        }
        imageProxy.close()
    }

    @OptIn(ExperimentalGetImage::class)
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val image = imageProxy.
        image ?: return null

        // Convert YUV to RGB
        val yBuffer = image.planes[0].buffer // Y
        val uBuffer = image.planes[1].buffer // U
        val vBuffer = image.planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val imageBytes = out.toByteArray()
        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        // Rotate the bitmap if necessary
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val matrix = Matrix()
        matrix.postRotate(rotationDegrees.toFloat())
        val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

        return rotatedBitmap
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(1 * MODEL_HEIGHT * MODEL_WIDTH * 3 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(MODEL_HEIGHT * MODEL_WIDTH)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until MODEL_HEIGHT) {
            for (j in 0 until MODEL_WIDTH) {
                val value = intValues[pixel++]
                val r = ((value shr 16) and 0xFF) / 255.0f
                val g = ((value shr 8) and 0xFF) / 255.0f
                val b = (value and 0xFF) / 255.0f
                val normalizedR = (r - 0.5f) * 2.0f
                val normalizedG = (g - 0.5f) * 2.0f
                val normalizedB = (b - 0.5f) * 2.0f
                byteBuffer.putFloat(normalizedR)
                byteBuffer.putFloat(normalizedG)
                byteBuffer.putFloat(normalizedB)
            }
        }
        return byteBuffer
    }

    private fun extractKeypointsFromHeatmap(
        heatmaps: Array<Array<Array<FloatArray>>>,
        overlayWidth: Float,
        overlayHeight: Float
    ): List<Pair<Float, Float>> {
        val keypoints = mutableListOf<Pair<Float, Float>>()

        val hmap_h = heatmaps[0].size
        val heatmapWidth = heatmaps[0][0].size

        val xScale = overlayWidth / heatmapWidth
        val yScale = overlayHeight / hmap_h

        val keypointIndices = (0 until NUM_KEYPOINTS).toList()  // Include all 17 keypoints

        val bodyConstraints = mutableMapOf<Int, Float>()  // Constraint map for y-coordinates

        for (index_key in keypointIndices) {
            var maxVal = Float.NEGATIVE_INFINITY
            var maxRow = 0
            var maxCol = 0

            for (row in 0 until hmap_h) {
                for (col in 0 until heatmapWidth) {
                    val score = heatmaps[0][row][col][index_key]
                    if (score > maxVal) {
                        maxVal = score
                        maxRow = row
                        maxCol = col
                    }
                }
            }

            // Apply confidence threshold
            if (maxVal >= CONFIDENCE_THRESHOLD) {
                val xPos = (maxCol + 0.5f) * xScale
                val yPos = (maxRow + 0.5f) * yScale

                // Apply skeletal constraints for knees
                if ((index_key == 13 || index_key == 14) &&
                    (bodyConstraints[11] != null || bodyConstraints[12] != null)) {
                    val hipY = if (bodyConstraints[11] != null) bodyConstraints[11]!! else bodyConstraints[12]!!
                    if (yPos < hipY) {
                        continue  // Knees should be below hips
                    }
                }

                // Update constraints for hips
                if (index_key == 11 || index_key == 12) {
                    bodyConstraints[index_key] = yPos
                }

                keypoints.add(Pair(xPos, yPos))
            }
        }

        // Log the number of keypoints detected
        Log.d("MainActivity", "Detected keypoints: ${keypoints.size}")
        keypoints.forEachIndexed { index, point ->
            Log.d("MainActivity", "Keypoint $index: (${point.first}, ${point.second})")
        }

        return keypoints
    }

    private fun drawKeypoints(keypoints: List<Pair<Float, Float>>) {
        // If keypoints are empty, skip processing
        if (keypoints.isEmpty()) return

        // Apply smoothing only if we have a previous frame's keypoints
        val smoothedKeypoints = if (previousKeypoints != null && previousKeypoints!!.size == keypoints.size) {
            keypoints.mapIndexed { index, (x, y) ->
                val (prevX, prevY) = previousKeypoints!![index]
                Pair(
                    prevX * SMOOTHING_FACTOR + x * (1 - SMOOTHING_FACTOR),
                    prevY * SMOOTHING_FACTOR + y * (1 - SMOOTHING_FACTOR)
                )
            }
        } else {
            keypoints // Use current keypoints if there's no previous frame for smoothing
        }

        // Update previous keypoints with smoothed values
        previousKeypoints = smoothedKeypoints
        overlayView.setKeyPoints(smoothedKeypoints)
        overlayView.postInvalidate()
    }
}
