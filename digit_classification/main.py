import pyaudio
import wave
import librosa

import numpy as np
import tflite_runtime.interpreter as tflite
from scipy import signal
import time
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas
import RPi.GPIO as GPIO

commands = ['three', 'six', 'seven', 'eight', 'four', 'two', 'one', 'five',
       'nine', 'zero']

def process_audio_data(waveform):
    """Process audio input.

    This function takes in raw audio data from a WAV file and does scaling
    and padding to 16000 length.

    """
    # if stereo, pick the left channel
    if len(waveform.shape) == 2:
        print("Stereo detected. Picking one channel.")
        waveform = waveform.T[1]
    else:
        waveform = waveform


    # normalise audio
    wabs = np.abs(waveform)
    wmax = np.max(wabs)
    waveform = waveform / wmax

    PTP = np.ptp(waveform)
    print("peak-to-peak: %.4f. Adjust as needed." % (PTP,))

    # return None if too silent
    if PTP < 0.5:
        return []

    # scale and center
    waveform = 2.0*(waveform - np.min(waveform))/PTP - 1

    # extract 16000 len (1 second) of data
    max_index = np.argmax(waveform)
    start_index = max(0, max_index-8000)
    end_index = min(max_index+8000, waveform.shape[0])
    waveform = waveform[start_index:end_index]

    waveform_padded = np.zeros((16000,))
    waveform_padded[:waveform.shape[0]] = waveform

    return waveform_padded

def get_spectrogram(waveform):

    waveform_padded = process_audio_data(waveform)

    if not len(waveform_padded):
        return []

    # compute spectrogram
    f, t, Zxx = signal.stft(waveform_padded, fs=16000, nperseg=255,
        noverlap = 124, nfft=256)
    # Output is complex, so take abs value
    spectrogram = np.abs(Zxx)
    return spectrogram

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

    # load TF Lite model
    interpreter = tflite.Interpreter(model_path="digit_audio_classification_lite.tflite")
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
    return commands[np.argmax(output_data[0])]
    # disp.show_txt(0, 12, commands[np.argmax(output_data[0])].upper(), True)

BUTTON_PIN = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN)
started=False
while True:
    if started==False:
        device = ssd1306(i2c(port=1, address=0x3c), width=128, height=64, rotate=0)
        # set the contrast to minimum.
        device.contrast(1)
        with canvas(device, dither=True) as draw:
                #draw.rectangle(device.bounding_box, outline='white', fill='black')
                
                message = "I am ready now"
                text_size = draw.textsize(message)
                draw.text((device.width - text_size[0], (device.height - text_size[1]) // 2), message, fill='white')
        started = True
    if GPIO.input(BUTTON_PIN):
        device = ssd1306(i2c(port=1, address=0x3c), width=128, height=64, rotate=0)
        # set the contrast to minimum.
        device.contrast(1)
        form_1 = pyaudio.paInt16 # 16-bit resolution
        chans = 1 # 1 channel
        samp_rate = 48000 # 48kHz sampling rate
        chunk = 4096 # 2^12 samples for buffer
        record_secs = 5 # seconds to record
        dev_index = 2 # device index found by p.get_device_info_by_index(ii)
        wav_output_filename = 'test1.wav' # name of .wav file

        audio = pyaudio.PyAudio() # create pyaudio instantiation

        # create pyaudio stream
        stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                            input_device_index = dev_index,input = True, \
                            frames_per_buffer=chunk)
        print("recording")
        with canvas(device, dither=True) as draw:
            #draw.rectangle(device.bounding_box, outline='white', fill='black')
            
            message = "recording"
            text_size = draw.textsize(message)
            draw.text((device.width - text_size[0], (device.height - text_size[1]) // 2), message, fill='white')
        ## give LED high here
        frames = []

        # loop through stream and append audio chunks to frame array
        for ii in range(0,int((samp_rate/chunk)*record_secs)):
            data = stream.read(chunk)
            frames.append(data)
        ## LED low here
        print("finished recording")
        with canvas(device, dither=True) as draw:
            #draw.rectangle(device.bounding_box, outline='white', fill='black')
            
            message = "finished recording"
            text_size = draw.textsize(message)
            draw.text((device.width - text_size[0], (device.height - text_size[1]) // 2), message, fill='white')

        # stop the stream, close it, and terminate the pyaudio instantiation
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # save the audio frames as .wav file
        wavefile = wave.open(wav_output_filename,'wb')
        wavefile.setnchannels(chans)
        wavefile.setsampwidth(audio.get_sample_size(form_1))
        wavefile.setframerate(samp_rate)
        wavefile.writeframes(b''.join(frames))
        wavefile.close()

        audio, sr = librosa.load('test1.wav', sr=16000)
        output = run_inference(audio)
        with canvas(device, dither=True) as draw:
            #draw.rectangle(device.bounding_box, outline='white', fill='black')
            
            message = output
            text_size = draw.textsize(message)
            draw.text((device.width - text_size[0], (device.height - text_size[1]) // 2), message, fill='white')

        # NB the display will be turn off after we exit this application.
        time.sleep(10)
        device.cleanup()
        started = False
        
        
