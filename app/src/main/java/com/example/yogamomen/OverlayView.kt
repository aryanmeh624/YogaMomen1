package com.example.yogamomen

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val keyPoints = mutableListOf<Pair<Float, Float>>()

    fun setKeyPoints(points: List<Pair<Float, Float>>) {
        keyPoints.clear()
        keyPoints.addAll(points)
        invalidate()
    }

    private val pointPaint = Paint().apply {
        color = Color.BLUE
        strokeWidth = 10f
        style = Paint.Style.FILL
    }

    private val linePaint = Paint().apply {
        color = Color.GREEN
        strokeWidth = 5f
        style = Paint.Style.STROKE
    }

    // Define pairs of keypoints to connect for skeleton
    private val skeletonConnections = listOf(
        Pair(0, 1), Pair(0, 2),
        Pair(1, 3), Pair(2, 4),
        Pair(5, 6), Pair(5, 7), Pair(7, 9),
        Pair(6, 8), Pair(8, 10),
        Pair(5, 11), Pair(6, 12),
        Pair(11, 12), Pair(11, 13), Pair(13, 15),
        Pair(12, 14), Pair(14, 16)
    )

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        // Draw keypoints
        for (point in keyPoints) {
            canvas.drawCircle(point.first, point.second, 10f, pointPaint)
        }

        // Draw skeleton lines
        for (connection in skeletonConnections) {
            if (connection.first < keyPoints.size && connection.second < keyPoints.size) {
                val start = keyPoints[connection.first]
                val end = keyPoints[connection.second]
                canvas.drawLine(start.first, start.second, end.first, end.second, linePaint)
            }
        }
    }
}
