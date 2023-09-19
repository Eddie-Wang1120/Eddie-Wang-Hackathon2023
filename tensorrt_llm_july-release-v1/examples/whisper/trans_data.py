from pathlib import Path
import shutil

def load_dataset(dataset_dir):
    label_file = None
    audio_file = []
    for file in dataset_dir.iterdir():
        if str(file).endswith('txt'):
            label_file = file
        else:
            audio_file.append(file)
    
    references = []
    with open(label_file, 'r') as f:
        for line in f:
            references.append((str(line).split(' ', 1))[1].replace('\n', ''))
    
    return audio_file, references

dataset_dir = Path('LibriSpeech/test-clean')
valid_dir = Path('LibriSpeech/valid-clean')
valid_file = valid_dir / 'valid.trans.txt'
valid_lines = []

with open(valid_file, 'w') as f:
    for dir in dataset_dir.iterdir():
        for child_dir in dir.iterdir():
            # audio_files, references = load_dataset(child_dir)
            # audio_files = sorted(audio_files)
            # shutil.copy(audio_files[0], valid_dir)
            for file in child_dir.iterdir():
                if str(file).endswith('txt'):
                    with open(file, 'r') as tf:
                        line = tf.readline()
                        valid_lines.append(line)
    
    valid_lines = sorted(valid_lines)
    f.writelines(valid_lines)