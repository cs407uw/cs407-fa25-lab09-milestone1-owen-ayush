package com.cs407.lab09

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.util.Log
import androidx.compose.ui.geometry.Offset
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

class BallViewModel : ViewModel() {

    private var ball: Ball? = null
    private var lastTimestamp: Long = 0L

    // Scale factor to convert sensor acceleration (m/s²) to pixel acceleration
    private val ACCELERATION_SCALE = 70f

    // Expose the ball's position as a StateFlow
    private val _ballPosition = MutableStateFlow(Offset.Zero)
    val ballPosition: StateFlow<Offset> = _ballPosition.asStateFlow()

    /**
     * Called by the UI when the game field's size is known.
     */
    fun initBall(fieldWidth: Float, fieldHeight: Float, ballSizePx: Float) {
        if (ball == null) {
            // Initialize the ball instance
            ball = Ball(fieldWidth, fieldHeight, ballSizePx)

            // Update the StateFlow with the initial position
            _ballPosition.value = Offset(ball!!.posX, ball!!.posY)
            Log.d("BallViewModel", "Ball initialized: fieldWidth=$fieldWidth, fieldHeight=$fieldHeight, ballSize=$ballSizePx, pos=(${ball!!.posX}, ${ball!!.posY})")
        }
    }

    /**
     * Called by the SensorEventListener in the UI.
     */
    fun onSensorDataChanged(event: SensorEvent) {
        // Ensure ball is initialized
        val currentBall = ball ?: run {
            Log.w("BallViewModel", "onSensorDataChanged called but ball is null!")
            return
        }

        if (event.sensor.type == Sensor.TYPE_GRAVITY) {
            if (lastTimestamp != 0L) {
                // Calculate the time difference (dT) in seconds
                val NS2S = 1.0f / 1000000000.0f
                val dT = (event.timestamp - lastTimestamp) * NS2S

                // Scale the acceleration values for pixel coordinates
                val xAcc = event.values[0] * ACCELERATION_SCALE
                val yAcc = -event.values[1] * ACCELERATION_SCALE

                Log.d("BallViewModel", "Sensor: x=$xAcc, y=$yAcc, dT=$dT, before pos=(${currentBall.posX}, ${currentBall.posY}), vel=(${currentBall.velocityX}, ${currentBall.velocityY})")

                // Update the ball's position and velocity
                // The sensor data represents acceleration opposite to gravity
                // Sensor x-axis matches screen x-axis, but sensor y-axis is inverted
                currentBall.updatePositionAndVelocity(
                    xAcc = xAcc,
                    yAcc = yAcc,
                    dT = dT
                )

                Log.d("BallViewModel", "After update: pos=(${currentBall.posX}, ${currentBall.posY}), vel=(${currentBall.velocityX}, ${currentBall.velocityY})")

                // Update the StateFlow to notify the UI
                _ballPosition.update { Offset(currentBall.posX, currentBall.posY) }
            }

            // Update the lastTimestamp
            lastTimestamp = event.timestamp
        }
    }

    fun reset() {
        // Reset the ball's state
        ball?.reset()

        // Update the StateFlow with the reset position
        ball?.let {
            _ballPosition.value = Offset(it.posX, it.posY)
        }

        // Reset the lastTimestamp
        lastTimestamp = 0L
    }
}