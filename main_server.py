# -*- coding: utf-8 -*-
# Configuracao de firewall para liberar trafego na porta do servidor
# run: iptables -I INPUT -p tcp --dport 5000 -j ACCEPT

"""
Created on Wed Jul 14 20:15:17 2019

@author: marcelo
"""

try:
    from flask import Flask, render_template, request, current_app, url_for
    import os
    from classifiers.binariza_image import image_classifier
    from classifiers.binariza_image import CLASSES_STR
    import cv2
    import numpy as np
except Exception as e:
    print(e)
    raise Exception(e)

app = Flask(__name__)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/results', methods=['GET', 'POST'])
def results(): #no caso de get, o nome da funcao deve ser o mesmo do template
    #nome = request.form.get('nome')
    #idade = request.form.get('idade')
    if request.method == 'POST':
        photo = request.files.get('photo')
        code_img = np.fromstring(photo.read(), np.uint8)
        decode_img = cv2.imdecode(code_img, cv2.IMREAD_COLOR)

        class_proba_image = image_classifier(decode_img)

        class_proba_image = np.round(100*class_proba_image, 2)

        classe = np.max(class_proba_image)
        print(classe)

        #photo.save()

        return render_template('results.html', class_image=class_proba_image, classes_str=CLASSES_STR, class_proba_choice=classe,filename=photo.filename)

    else:
        tipo = 'Nenhuma imagem foi carregada!'
        return render_template('error.html', tipo=tipo)

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=True, use_reloader=True)

