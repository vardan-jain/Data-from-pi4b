def run_inference(waveform):

    # get spectrogram data
    spectrogram = get_spectrogram(waveform)

    if not len(spectrogram):
        #disp.show_txt(0, 0, "Silent. Skip...", True)
        print("Too silent. Skipping...")
        #time.sleep(1)
        return

    spectrogram1= np.reshape(spectrogram,
                (-1, spectrogram.shape[0], spectrogram.shape[1], 1))

    if VERBOSE_DEBUG:
        print("spectrogram1: %s, %s, %s" % (type(spectrogram1),
               spectrogram1.dtype, spectrogram1.shape))

    # load TF Lite model
    interpreter = tf.lite.Interpreter(model_path="digit_audio_classification_lite.tflite")
    # interpreter = tf.lite.Interpreter(model_path="./drive/MyDrive/uaspeech/digit_audio_classification_spectrogram.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #print(input_details)
    #print(output_details)

    input_shape = input_details[0]['shape']
    print(input_shape)
    input_data = spectrogram1.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    print("running inference...")
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    if VERBOSE_DEBUG:
        print(output_data[0])
    print(">>> "+commands[np.argmax(output_data[0])])
    # disp.show_txt(0, 12, commands[np.argmax(output_data[0])].upper(), True)