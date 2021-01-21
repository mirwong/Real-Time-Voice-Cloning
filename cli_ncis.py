from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from utils.modelutils import check_model_paths
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import argparse
import torch
import sys
from audioread.exceptions import NoBackendError

if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path, 
                        default="pretrained/encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_dir", type=Path, 
                        default="pretrained/synthesizer/saved_models/logs-pretrained/",
                        help="Directory containing the synthesizer model")
    parser.add_argument("-v", "--voc_model_fpath", type=Path, 
                        default="pretrained/vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--low_mem", action="store_true", help=\
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    parser.add_argument("--no_mp3_support", action="store_true", help=\
        "If True, disallows loading mp3 files to prevent audioread errors when ffmpeg is not installed.")
    args = parser.parse_args()
    print_args(args, parser)
    if not args.no_sound:
        import sounddevice as sd

    if not args.no_mp3_support:
        try:
            librosa.load("samples/1320_00000.mp3")
        except NoBackendError:
            print("Librosa will be unable to open mp3 files if additional software is not installed.\n"
                  "Please install ffmpeg or add the '--no_mp3_support' option to proceed without support for mp3 files.")
            exit(-1)
        
    print("Running a test of your configuration...\n")
        
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Print some environment information (for debugging purposes)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" % 
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")
    
    ## Remind the user to download pretrained models if needed
    check_model_paths(encoder_path=args.enc_model_fpath, synthesizer_path=args.syn_model_dir,
                      vocoder_path=args.voc_model_fpath)
    
    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=args.low_mem, seed=args.seed)
    vocoder.load_model(args.voc_model_fpath)
    
    
    ## Interactive speech generation
    print("This is a GUI-less example of interface to SV2TTS. The purpose of this script is to "
          "show how you can interface this project easily with your own. See the source code for "
          "an explanation of what is happening.\n")
    
    try:
        # Get the reference audio filepath
        # message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
        #           "wav, m4a, flac, ...):\n"
        # in_fpath = Path(input(message).replace("\"", "").replace("\'", ""))

        # if in_fpath.suffix.lower() == ".mp3" and args.no_mp3_support:
        #     print("Can't Use mp3 files please try again:")
        #     continue
        ## Computing the embedding
        # First, we load the wav using the function that the speaker encoder provides. This is 
        # important: there is preprocessing that must be applied.

        in_paths = ["../voices/abby_all.wav"]
        embeds = []

        for in_fpath in in_paths:
            # The following two methods are equivalent:
            # - Directly load from the filepath:
            preprocessed_wav = encoder.preprocess_wav(in_fpath)
            # - If the wav is already loaded:
            original_wav, sampling_rate = librosa.load(str(in_fpath))
            preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
            print("Loaded file succesfully")
            
            # Then we derive the embedding. There are many functions and parameters that the 
            # speaker encoder interfaces. These are mostly for in-depth research. You will typically
            # only use this function (with its default parameters):
            embed = encoder.embed_utterance(preprocessed_wav)
            print("Created the embedding for {}".format(in_fpath))
            embeds.append(embed)

        ## Generating the spectrogram
        print("Generating spectrogram")
        # text = input("Write a sentence (+-20 words) to be synthesized:\n")
        text = "I looked at the materials the Port to Port killer used and matched those with people who had purchased three or more of those materials in the past week."
        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        texts = [text]
        # If you know what the attention layer alignments are, you can retrieve them here by
        # passing return_alignments=True
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        print("Created the mel spectrogram")
        
        
        ## Generating the waveform
        print("Synthesizing the waveform:")

        # If seed is specified, reset torch seed and reload vocoder
        if args.seed is not None:
            torch.manual_seed(args.seed)
            vocoder.load_model(args.voc_model_fpath)

        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        generated_wav = vocoder.infer_waveform(spec)
        
        
        ## Post-generation
        # There's a bug with sounddevice that makes the audio cut one second earlier, so we
        # pad it.
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

        # Trim excess silences to compensate for gaps in spectrograms (issue #53)
        generated_wav = encoder.preprocess_wav(generated_wav)
        
        # # Play the audio (non-blocking)
        # if not args.no_sound:
        #     try:
        #         sd.stop()
        #         sd.play(generated_wav, synthesizer.sample_rate)
        #     except sd.PortAudioError as e:
        #         print("\nCaught exception: %s" % repr(e))
        #         print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
        #     except:
        #         raise
            
        # Save it on the disk
        filename = "../voices/demo_output_%02d.wav" % num_generated
        print(generated_wav.dtype)
        sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
        num_generated += 1
        print("\nSaved output as %s\n\n" % filename)
            
            
    except Exception as e:
        print("Caught exception: %s" % repr(e))
        print("Restarting\n")
