import re
from collections import defaultdict
import os
import codecs

def extract_text(line):
    pattern = r'<Sync time="(\d+\.\d+)"/>(.*?)<Sync time="(\d+\.\d+)"/>'
    matches = re.findall(pattern, line, re.DOTALL)
    text_segments = [text.strip() for start_time, text, end_time in matches if text.strip()]
    return ' '.join(text_segments)

import os
from collections import defaultdict
import xml.etree.ElementTree as ET

import os
from collections import defaultdict
import xml.etree.ElementTree as ET

import os
from collections import defaultdict
import xml.etree.ElementTree as ET

import os
from collections import defaultdict
import xml.etree.ElementTree as ET

def parse_xml_files(file_paths):
    adult_turns = []
    child_turns = []

    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                content_bytes = file.read()

            root = ET.fromstring(content_bytes)

            # Determine the adult speaker ID
            adult_speaker_id = None
            for speaker in root.findall('Speakers/Speaker'):
                if speaker.get('name') == 'Adulte':
                    adult_speaker_id = speaker.get('id').split('spk')[1]
                    break

            for turn in root.iter('Turn'):
                speaker_id = turn.get('speaker')
                if speaker_id:
                    speaker_id = speaker_id.split('spk')[1]

                text_segments = []
                for sync in turn.iter('Sync'):
                    text = sync.tail.strip() if sync.tail else ''
                    text_segments.append(text)

                if text_segments:
                    turn_text = ' '.join(text_segments)
                    turn_words = turn_text.split()

                    if speaker_id == adult_speaker_id:
                        adult_turns.append(turn_words)
                    else:
                        child_turns.append(turn_words)

    return adult_turns, child_turns
file_paths = ['/Users/quinncoleman/Downloads/TCOF Gaelle1 Sow.trs', '/Users/quinncoleman/Downloads/TCOF Hugo1 Bar.trs', '/Users/quinncoleman/Downloads/tcof/13/Corpus/Enfants/Corpus longitudinaux/hector_aub (5.6-6.0)/hector1_aub/hector1_aub.trs']

adult_turns, child_turns = parse_xml_files(file_paths)
print(adult_turns)