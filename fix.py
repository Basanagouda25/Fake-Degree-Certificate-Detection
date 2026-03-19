import json

nb_path = r'c:\Users\basan\Desktop\project\Minor_Project\Fake_certificate_detection.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f: 
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        src = cell.get('source', [])
        for i in range(len(src)):
            src[i] = src[i].replace('base_model.trainable = True', 'base_model.trainable = False')
            if 'for layer in base_model.layers' in src[i] and not src[i].lstrip().startswith('#'): 
                src[i] = '# ' + src[i]
            if 'layer.trainable = False' in src[i] and 'base_model' not in src[i] and not src[i].lstrip().startswith('#'): 
                src[i] = '# ' + src[i]
            src[i] = src[i].replace('learning_rate=0.00003', 'learning_rate=0.001')
            src[i] = src[i].replace("Dense(1, activation='sigmoid')(x)", "Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)")

with open(nb_path, 'w', encoding='utf-8') as f: 
    json.dump(nb, f, indent=1)
