package com.cs498.portmanteau.gl;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

/**
 * Uses the sensor API to determine the phones orientation.
 * Registering for events from the accelerator and the magnetometer (compass)
 * a rotation matrix is computed. This matrix can be used to rotate an
 * OpenGL scene.
 */
public class PhoneOrientation {
	private SensorManager sensorMan;
	private Sensor sensorAcce;
	private Sensor sensorMagn;
	private SensorEventListener listener;
	private float matrix[]=new float[16];
	private float armatrix[]=new float[16];
	static final float SMOOTH_OUT_ACCEL = 2f;
	static final float SMOOTH_IN_ACCEL = 0.33334f;
	static final float SMOOTH_OUT_ORIE = 1f;
	static final float SMOOTH_IN_ORIE = 0.5f;
	
	public PhoneOrientation() {
	}

	public void start(Context context) {
        listener = new SensorEventListener() {
        	private float orientation[]=new float[3];
        	private float acceleration[]=new float[3];

        	public void onAccuracyChanged(Sensor arg0, int arg1){}

        	public void onSensorChanged(SensorEvent evt) {
        		int type=evt.sensor.getType();
        		
        		//Smoothing the sensor data a bit seems like a good idea.
        		if (type == Sensor.TYPE_MAGNETIC_FIELD) {
        			orientation[0]=(orientation[0]*SMOOTH_OUT_ORIE+evt.values[0])*SMOOTH_IN_ORIE;
        			orientation[1]=(orientation[1]*SMOOTH_OUT_ORIE+evt.values[1])*SMOOTH_IN_ORIE;
        			orientation[2]=(orientation[2]*SMOOTH_OUT_ORIE+evt.values[2])*SMOOTH_IN_ORIE;
        		} else if (type == Sensor.TYPE_ACCELEROMETER) {
       				acceleration[0]=(acceleration[0]*SMOOTH_OUT_ACCEL+evt.values[0])*SMOOTH_IN_ACCEL;
       				acceleration[1]=(acceleration[1]*SMOOTH_OUT_ACCEL+evt.values[1])*SMOOTH_IN_ACCEL;
       				acceleration[2]=(acceleration[2]*SMOOTH_OUT_ACCEL+evt.values[2])*SMOOTH_IN_ACCEL;
        		}
        		if ((type==Sensor.TYPE_MAGNETIC_FIELD) || (type==Sensor.TYPE_ACCELEROMETER)) {
        			float newMat[]=new float[16];
        			if (SensorManager.getRotationMatrix(newMat, null, acceleration, orientation)) {
//        				SensorManager.remapCoordinateSystem(newMat, SensorManager.AXIS_Y, SensorManager.AXIS_X, matrix);

//        				SensorManager.remapCoordinateSystem(newMat, SensorManager.AXIS_MINUS_Z, SensorManager.AXIS_MINUS_Y, armatrix);
//        				SensorManager.remapCoordinateSystem(armatrix, SensorManager.AXIS_MINUS_Z, SensorManager.AXIS_X, armatrix);
//        				armatrix = newMat;
//        				SensorManager.remapCoordinateSystem(newMat, SensorManager.AXIS_Y, SensorManager.AXIS_Z, armatrix);

        			
        			
        				SensorManager.remapCoordinateSystem(newMat, SensorManager.AXIS_Y, SensorManager.AXIS_MINUS_X, armatrix);
//        				armatrix = newMat;
        			}
        		}
        	}
        };

        sensorMan = (SensorManager)context.getSystemService(Context.SENSOR_SERVICE);
		sensorAcce = sensorMan.getSensorList(Sensor.TYPE_ACCELEROMETER).get(0);
		sensorMagn = sensorMan.getSensorList(Sensor.TYPE_MAGNETIC_FIELD).get(0);
		
		sensorMan.registerListener(listener, sensorAcce, SensorManager.SENSOR_DELAY_GAME);
		sensorMan.registerListener(listener, sensorMagn, SensorManager.SENSOR_DELAY_GAME);		
	}
	
	public float[] getMatrix() {
//		Log.d("FoneOrientati", "Printing matrix");
//		printMatrix(matrix);
		return matrix;
	}
	
	public float[] getARMatrix() {
//		Log.d("FoneOrientati", "Printing armatrix");
//		printMatrix(armatrix);
		return armatrix;
	}
	
	public void printMatrix(float[] matrix) {
		String outstr = "matri: \n";
		for (int i=0; i<matrix.length;i++) {
			outstr += matrix[i]+",";
			if (i % 4 == 3) {
				outstr += "\n";
			}
		}
		Log.d("FoneOrientati", outstr);
	}
	

	public void finish() {
		sensorMan.unregisterListener(listener);
	}
}

