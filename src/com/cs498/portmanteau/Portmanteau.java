package com.cs498.portmanteau;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedList;


import android.app.Activity;
import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.ViewGroup.LayoutParams;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.LinearLayout;

import com.opencv.camera.NativePreviewer;
import com.opencv.camera.NativeProcessor;
import com.opencv.camera.NativeProcessor.PoolCallback;
import com.opencv.jni.image_pool;
import com.theveganrobot.cvcamera.jni.Processor;
import com.theveganrobot.cvcamera.jni.cvcamera;

public class Portmanteau extends Activity {
	/**
	 * Avoid that the screen get's turned off by the system.
	 */
	public void disableScreenTurnOff() {
		getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
				WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
	}

	/**
	 * Set's the orientation to landscape, as this is needed by AndAR.
	 */
	public void setOrientation() {
		setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
	}

	/**
	 * Maximize the application.
	 */
	public void setFullscreen() {
		requestWindowFeature(Window.FEATURE_NO_TITLE);
		getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
				WindowManager.LayoutParams.FLAG_FULLSCREEN);
	}

	public void setNoTitle() {
		requestWindowFeature(Window.FEATURE_NO_TITLE);
	}

	private NativePreviewer mPreview;
	private GLPortmanteauCameraViewer glview;


	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		setFullscreen();
		disableScreenTurnOff();

		FrameLayout frame = new FrameLayout(this);

		// Create our Preview view and set it as the content of our activity.
		mPreview = new NativePreviewer(getApplication(), 800, 600);
		//mPreview
		
		setupTestImage();

		LayoutParams params = new LayoutParams(LayoutParams.WRAP_CONTENT,
				LayoutParams.WRAP_CONTENT);
		params.height = getWindowManager().getDefaultDisplay().getHeight();
		params.width = (int) (params.height * 4.0 / 2.88);

		setupVideoOverlay(frame, params);

		// make the glview overlay ontop of video preview
		mPreview.setZOrderMediaOverlay(false);

		setupGLView(frame, params);

		
//		LinearLayout buttons = setupCaptureButton();
//
//		setupFocusButton(buttons);

//		frame.addView(buttons);
		setContentView(frame);
	}
	
	private void setupTestImage() {
			final File dir = this.getFilesDir();
			final File imageFile = new File(dir, "IMAG0265.jpg");
			try {
				IoUtils.extractZipResource(this.getResources().openRawResource(R.raw.testfiles), dir, true);
			} catch (final IOException e) {
				e.printStackTrace();
			}
		System.out.println(" Calling processorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr");
		processor.setupDescriptorExtractorMatcher(imageFile.getAbsolutePath(), cvcamera.DETECT_SURF);
	}
	/**
	 * Extract a resource into a real file
	 * 
	 * @param in typically given as getResources().openRawResource(R.raw.something)
	 * @param name of the resulting file
	 * @param directory target directory
	 * @return the resulting file
	 * @throws IOException
	 */
	public static File extractResource(InputStream in, String filename, File directory) throws IOException {
		int n = in.available();
		byte[] buffer = new byte[n];
		in.read(buffer);
		in.close();
		File file = new File(directory, filename);
		FileOutputStream out = new FileOutputStream(file);
		out.write(buffer);
		out.close();
		return file;
	}

	private void setupVideoOverlay(FrameLayout frame, LayoutParams params) {
		LinearLayout vidlay = new LinearLayout(getApplication());

		vidlay.setGravity(Gravity.CENTER);
		vidlay.addView(mPreview, params);
		frame.addView(vidlay);
	}

	private void setupGLView(FrameLayout frame, LayoutParams params) {
		glview = new GLPortmanteauCameraViewer(getApplication(), false, 0, 0);
		glview.setZOrderMediaOverlay(true);

		LinearLayout gllay = new LinearLayout(getApplication());

		gllay.setGravity(Gravity.CENTER);
		gllay.addView(glview, params);
		frame.addView(gllay);
	}

	private LinearLayout setupCaptureButton() {
		ImageButton capture_button = new ImageButton(getApplicationContext());
		capture_button.setImageDrawable(getResources().getDrawable(
				android.R.drawable.ic_menu_camera));
		capture_button.setLayoutParams(new LayoutParams(
				LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT));
		capture_button.setOnClickListener(new View.OnClickListener() {

			@Override
			public void onClick(View v) {
				//do capture thing

			}
		});

		LinearLayout buttons = new LinearLayout(getApplicationContext());
		buttons.setLayoutParams(new LayoutParams(LayoutParams.WRAP_CONTENT,
				LayoutParams.WRAP_CONTENT));

		buttons.addView(capture_button);
		return buttons;
	}

	private void setupFocusButton(LinearLayout buttons) {
		Button focus_button = new Button(getApplicationContext());
		focus_button.setLayoutParams(new LayoutParams(
				LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT));
		focus_button.setText("Focus");
		focus_button.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				mPreview.postautofocus(100);
			}
		});
		buttons.addView(focus_button);
	}


	@Override
	protected void onPause() {
		super.onPause();

		// clears the callback stack
		mPreview.onPause();

		glview.onPause();

	}

	@Override
	protected void onResume() {
		super.onResume();
		glview.onResume();
		mPreview.setParamsFromPrefs(getApplicationContext());
		// add an initial callback stack to the preview on resume...
		// this one will just draw the frames to opengl
		LinkedList<NativeProcessor.PoolCallback> cbstack = new LinkedList<PoolCallback>();
		cbstack.add(glview.getDrawCallback());
		cbstack.addFirst(new SURFProcessor());

		mPreview.addCallbackStack(cbstack);
		mPreview.onResume();
		
	}

	
	
	// final processor so taht these processor callbacks can access it
	final Processor processor = new Processor();

	class SURFProcessor implements NativeProcessor.PoolCallback {

		@Override
		public void process(int idx, image_pool pool, long timestamp,
				NativeProcessor nativeProcessor) {
			processor.detectAndDrawFeatures(idx, pool, cvcamera.DETECT_SURF);

		}

	}

}
