import re
from collections import defaultdict
import os
import codecs
import xml.etree.ElementTree as ET 
from transformers import FlaubertTokenizer
from sacremoses import MosesTokenizer, MosesDetokenizer
from sklearn.model_selection import train_test_split 



# class preprocessing():
#     def __init__(self):
#         super().__init__()
        
#         self.tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")

        
def extract_text(line):
    pattern = r'<Sync time="(\d+\.\d+)"/>(.*?)<Sync time="(\d+\.\d+)"/>'
    matches = re.findall(pattern, line, re.DOTALL)
    text_segments = [text.strip() for start_time, text, end_time in matches if text.strip()]
    return ' '.join(text_segments)



# def parse_xml_files( file_paths):
#     tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")
        
#     adult_turns = []
#     child_turns = []

#     for file_path in file_paths:
#         if os.path.exists(file_path):
#             with open(file_path, 'rb') as file:
#                 content_bytes = file.read()

#                 root = ET.fromstring(content_bytes)

#                 # Determine the adult speaker ID
#                 adult_speaker_id = None
#                 for speaker in root.findall('Speakers/Speaker'):
#                     if speaker.get('name') == 'Adulte':
#                         adult_speaker_id = speaker.get('id').split('spk')[1]
#                         break

#                 for turn in root.iter('Turn'):
#                     speaker_id = turn.get('speaker')
#                     if speaker_id:
#                         speaker_id = speaker_id.split('spk')[1]

#                     text_segments = []
#                     for sync in turn.iter('Sync'):
#                         text = sync.tail.strip() if sync.tail else ''
#                         text_segments.append(text)

#                     if text_segments:
#                         turn_text = ' '.join(text_segments)

#                         if speaker_id == adult_speaker_id:
#                             adult_turns.append(turn_text)
#                         else:
#                             child_turns.append(turn_text)
                            
#     child_tokenized = tokenizer(child_turns)
#     adult_tokenized = tokenizer(adult_turns)
#     print(len(child_turns))
#     print(len(adult_turns))
#     X_train, X_test, y_train, y_test = train_test_split(child_turns,adult_turns ,shuffle=True) 
#     X_train = tokenizer(X_train)
#     X_test = tokenizer(X_test)
#     y_train = tokenizer(y_train)
#     y_test = tokenizer(y_test)



    

#     return (X_train, X_test, y_train, y_test)
def parse_xml_files(file_paths):
    tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")

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

            turns = root.iter('Turn')
            adult_turn = next(turns, None)
            child_turn = next(turns, None)

            while adult_turn is not None and child_turn is not None:
                adult_text_segments = []
                for sync in adult_turn.iter('Sync'):
                    text = sync.tail.strip() if sync.tail else ''
                    adult_text_segments.append(text)

                child_text_segments = []
                for sync in child_turn.iter('Sync'):
                    text = sync.tail.strip() if sync.tail else ''
                    child_text_segments.append(text)

                if adult_text_segments and child_text_segments:
                    adult_turn_text = ' '.join(adult_text_segments)
                    adult_turns.append(adult_turn_text)

                    child_turn_text = ' '.join(child_text_segments)
                    child_turns.append(child_turn_text)

                adult_turn = next(turns, None)
                child_turn = next(turns, None)
    child_tokenized = tokenizer(child_turns)
    adult_tokenized = tokenizer(adult_turns)
    #print(len(child_turns))
    #print(len(adult_turns))
    X_train, X_test, y_train, y_test = train_test_split(child_turns,adult_turns ,shuffle=True) 
    X_train = tokenizer(X_train,padding=True,truncation=True, return_tensors="tf")
    X_test = tokenizer(X_test,padding=True, truncation=True,return_tensors="tf")
    y_train = tokenizer(y_train,padding=True, truncation=True,return_tensors="tf")
    y_test = tokenizer(y_test,padding=True, truncation=True,return_tensors="tf")
    print(X_train['input_ids'].shape)

    return (X_train, X_test, y_train, y_test, tokenizer)
file_paths = ['data/adrien1_bia.trs', 'data/adrien2_bia.trs', 'data/adrien3_bia.trs', 
              'data/Akoub15_Can_Anon.trs', 'data/Akoub16_Can_Anon.trs', 
              'data/Alhem1_Can_Anon.trs', 'data/Alhem2_Can_Anon.trs', 'data/Alhem3_Can_Anon.trs', 'data/Alhem4_Can_Anon.trs', 'data/Cassandra11_Can_Anon.trs', 'data/Cassandra12_Can_Anon.trs', 'data/celia1_can.trs', 'data/celia1_gav.trs', 'data/celia2_can.trs', 'data/celia2_gav.trs', 'data/celia3_can.trs', 'data/celia3_gav.trs', 'data/celia4_can.trs', 'data/celia5_can.trs', 'data/celia6_can.trs', 'data/celia7_can.trs', 'data/celia8_can.trs', 'data/celia9_can.trs', 'data/celia10_can.trs', 'data/celia11_can.trs', 'data/celia12_can.trs', 'data/fanny1_cha.trs', 'data/fanny2_cha.trs', 'data/fanny3_cha.trs', 'data/Ferdinand3_Can_Anon.trs', 'data/Ferdinand4_Can_Anon.trs', 'data/gaelle1_sow.trs', 'data/gaelle2_sow.trs', 'data/gaelle3_sow.trs', 'data/gaelle4_sow.trs', 'data/gaelle5_sow.trs', 'data/gaelle6_sow.trs', 'data/garance1_sow.trs', 'data/garance2_sow.trs', 'data/garance3_sow.trs', 'data/garance4_sow.trs', 'data/garance5_sow.trs', 'data/garance6_sow.trs', 'data/hector1_aub.trs', 'data/hector2_aub.trs', 'data/hector3_aub.trs', 'data/hugo1_bar.trs', 'data/hugo2_bar.trs', 'data/hugo3_bar.trs', 'data/india1_bru.trs', 'data/india2_bru.trs', 'data/india3_bru.trs', 'data/lionel1_can.trs', 'data/lionel2_can.trs', 'data/lionel3_can.trs', 'data/lionel4_can.trs', 'data/lionel5_can.trs', 'data/lionel6_can.trs', 'data/lionel7_can.trs', 'data/lionel8_can.trs', 'data/lionel9_can.trs', 'data/louise1_sow.trs', 'data/louise2_sow.trs', 'data/louise3_sow.trs', 'data/louise4_sow.trs', 'data/louise5_sow.trs', 'data/louise6_sow.trs', 'data/lucie1_can.trs', 'data/lucie2_can.trs', 'data/lucie3_can.trs', 'data/lucie4_can.trs', 'data/lucie5_can.trs', 'data/lucie6_can.trs', 'data/lucie7_can.trs', 'data/lucie8_can.trs', 'data/lucille1_cha.trs', 'data/lucille2_cha.trs', 'data/lucille3_cha.trs', 'data/maelle1_rou.trs', 'data/maelle2_rou.trs', 'data/maelle3_rou.trs', 'data/marie1_gue.trs', 'data/marie2_gue.trs', 'data/marie3_gue.trs', 'data/sarah1_can.trs', 'data/sarah2_can.trs', 'data/sarah3_can.trs', 'data/sarah4_can.trs', 'data/sarah5_can.trs', 'data/sarah6_can.trs', 'data/sarah7_can.trs', 'data/sarah8_can.trs', 'data/sarah9_can.trs', 'data/sarah10_can.trs', 'data/sarah11_can.trs', 'data/sarah12_can.trs', 'data/valentine1_sow.trs', 'data/valentine2_sow.trs', 'data/valentine3_sow.trs', 'data/valentine4_sow.trs', 'data/valentine5_sow.trs', 'data/valentine6_sow.trs', 'data/vincent1_can.trs', 'data/vincent2_can.trs', 'data/vincent3_can.trs', 'data/vincent4_can.trs', 'data/vincent5_can.trs', 'data/vincent6_can.trs', 'data/vincent7_can.trs', 'data/vincent8_can.trs', 'data/vincent9_can.trs', 'data/vincent10_can.trs', 'data/vincent11_can.trs', 'data/vincent12_can.trs', 'data/vincent12_can.trs', 'data/vincent13_can.trs', 'data/vincent14_can.trs', 'data/vincent15_can.trs', 'data/vincent16_can.trs', 'data/vincent17_can.trs', 'data/vincent18_can.trs', 'data/vincent19_can.trs', 'data/vincent20_can.trs', 'data/vincent21_can.trs', 'data/vincent22_can.trs']
# adult_turns, child_turns = parse_xml_files(file_paths)
# print(len(adult_turns))
# print(len(child_turns))

#X_train, X_test, y_train, y_test, tokenizer = parse_xml_files(file_paths)
#print(X_test['input_ids'])
