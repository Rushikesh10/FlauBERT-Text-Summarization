import numpy as np
from Summarizer import Summarizer
from flask import Flask, request, render_template

def summarize(model, method, text, nb_sentences):
    summarizer = Summarizer()
    summarizer.init_model(model, log=True)
    summarizer.fit(text)
    if method == 'mean':
        summary = summarizer.mean_similarity_summary(nb_sentences=nb_sentences)
    return summary

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page

def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            text = str(request.form['line'])
            def load_preprocess_text(text):
                text = text.split('.')
                return np.array(text)

            text = load_preprocess_text(text)
            summary = summarize('flaubert','mean',text,5)
            l = []
            for sentence in summary:
                print(sentence)
                l.append(sentence)

            s = " ".join(l)


            # showing the prediction results in a UI
            return render_template('results.html',prediction=str(s))
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app


