from preprocessing import parse_xml_files
import tensorflow as tf
from encoderdecoder import Encoderdecoder
import numpy as np

def train(model, train_inputs, train_labels):
    batch_num = train_inputs['input_ids'].shape[0] // 48
    for i in range(batch_num):
        input = train_inputs[i*48:i*48+48]
        label = train_labels[i*48:i*48+48]
        with tf.GradientTape() as tape:
            loss = model(input['input_ids'], label['input_ids'], training=True)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def test(model, test_inputs, test_labels):
    batch_num = test_inputs['input_ids'].shape[0] // 32
    for i in range(batch_num):
          input = test_inputs[i*32:i*32+32]
          label = test_labels[i*32:i*32+32]
          loss = model(input['input_ids'], label['input_ids'], training=True)
    return loss

def main():
    file_paths = ['data/adrien1_bia.trs', 'data/adrien2_bia.trs', 'data/adrien3_bia.trs', 
              'data/Akoub15_Can_Anon.trs', 'data/Akoub16_Can_Anon.trs', 
              'data/Alhem1_Can_Anon.trs', 'data/Alhem2_Can_Anon.trs', 'data/Alhem3_Can_Anon.trs', 'data/Alhem4_Can_Anon.trs', 'data/Cassandra11_Can_Anon.trs', 'data/Cassandra12_Can_Anon.trs']#, 'data/celia1_can.trs', 'data/celia1_gav.trs', 'data/celia2_can.trs', 'data/celia2_gav.trs', 'data/celia3_can.trs', 'data/celia3_gav.trs', 'data/celia4_can.trs', 'data/celia5_can.trs', 'data/celia6_can.trs', 'data/celia7_can.trs', 'data/celia8_can.trs', 'data/celia9_can.trs', 'data/celia10_can.trs', 'data/celia11_can.trs', 'data/celia12_can.trs', 'data/fanny1_cha.trs', 'data/fanny2_cha.trs', 'data/fanny3_cha.trs', 'data/Ferdinand3_Can_Anon.trs', 'data/Ferdinand4_Can_Anon.trs', 'data/gaelle1_sow.trs', 'data/gaelle2_sow.trs', 'data/gaelle3_sow.trs', 'data/gaelle4_sow.trs', 'data/gaelle5_sow.trs', 'data/gaelle6_sow.trs', 'data/garance1_sow.trs', 'data/garance2_sow.trs', 'data/garance3_sow.trs', 'data/garance4_sow.trs', 'data/garance5_sow.trs', 'data/garance6_sow.trs', 'data/hector1_aub.trs', 'data/hector2_aub.trs', 'data/hector3_aub.trs', 'data/hugo1_bar.trs', 'data/hugo2_bar.trs', 'data/hugo3_bar.trs', 'data/india1_bru.trs', 'data/india2_bru.trs', 'data/india3_bru.trs', 'data/lionel1_can.trs', 'data/lionel2_can.trs', 'data/lionel3_can.trs', 'data/lionel4_can.trs', 'data/lionel5_can.trs', 'data/lionel6_can.trs', 'data/lionel7_can.trs', 'data/lionel8_can.trs', 'data/lionel9_can.trs', 'data/louise1_sow.trs', 'data/louise2_sow.trs', 'data/louise3_sow.trs', 'data/louise4_sow.trs', 'data/louise5_sow.trs', 'data/louise6_sow.trs', 'data/lucie1_can.trs', 'data/lucie2_can.trs', 'data/lucie3_can.trs', 'data/lucie4_can.trs', 'data/lucie5_can.trs', 'data/lucie6_can.trs', 'data/lucie7_can.trs', 'data/lucie8_can.trs', 'data/lucille1_cha.trs', 'data/lucille2_cha.trs', 'data/lucille3_cha.trs', 'data/maelle1_rou.trs', 'data/maelle2_rou.trs', 'data/maelle3_rou.trs', 'data/marie1_gue.trs', 'data/marie2_gue.trs', 'data/marie3_gue.trs', 'data/sarah1_can.trs', 'data/sarah2_can.trs', 'data/sarah3_can.trs', 'data/sarah4_can.trs', 'data/sarah5_can.trs', 'data/sarah6_can.trs', 'data/sarah7_can.trs', 'data/sarah8_can.trs', 'data/sarah9_can.trs', 'data/sarah10_can.trs', 'data/sarah11_can.trs', 'data/sarah12_can.trs', 'data/valentine1_sow.trs', 'data/valentine2_sow.trs', 'data/valentine3_sow.trs', 'data/valentine4_sow.trs', 'data/valentine5_sow.trs', 'data/valentine6_sow.trs', 'data/vincent1_can.trs', 'data/vincent2_can.trs', 'data/vincent3_can.trs', 'data/vincent4_can.trs', 'data/vincent5_can.trs', 'data/vincent6_can.trs', 'data/vincent7_can.trs', 'data/vincent8_can.trs', 'data/vincent9_can.trs', 'data/vincent10_can.trs', 'data/vincent11_can.trs', 'data/vincent12_can.trs', 'data/vincent12_can.trs', 'data/vincent13_can.trs', 'data/vincent14_can.trs', 'data/vincent15_can.trs', 'data/vincent16_can.trs', 'data/vincent17_can.trs', 'data/vincent18_can.trs', 'data/vincent19_can.trs', 'data/vincent20_can.trs', 'data/vincent21_can.trs', 'data/vincent22_can.trs']
    X_train, X_test, y_train, y_test, tokenizer = parse_xml_files(file_paths)


    model = Encoderdecoder(tokenizer)
    for epoch in range(4):
        loss = train(model, X_train, y_train)
        perp = np.exp(loss)

        print(f"Epoch {epoch+1} - Loss: {loss:.4f} - Perplexity: {perp:.4f}")

    loss = test(model, X_test, y_test)
    perp = np.exp(loss)

    print(f"Test - Loss: {loss:.4f} - Perplexity: {perp:.4f}")
    

if __name__ == '__main__':
    main()
