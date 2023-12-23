import flask.wrappers
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from inference import Inference


application = Flask(__name__, template_folder='templates')

# infer = Inference()


@application.route('/')
def index():
    return render_template('index.html')


@application.route('/ganerate.html')
def ganerate_html():
    return render_template('ganerate.html')


@application.route('/ganerate', methods=['POST'])
def gan():
    dataForm = request.form
    prompt_ru = infer.generate_prompt(author=dataForm['author'], painting_name=dataForm['NamePicture'])
    prompt_en = infer.translate(text=prompt_ru)
    infer.generate_img(prompt=prompt_ru)
    return render_template('ganerate.html',
                           author=dataForm['author'],
                           prompt_en=prompt_en, prompt_ru=prompt_ru)


if __name__ == '__main__':
    application.run(debug=True)