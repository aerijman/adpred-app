import flask
from bin.utils import *
import requests
import re, os

app = flask.Flask(__name__)


# Index page
@app.route('/', methods=['GET','POST'])
def index():

    if flask.request.method == 'GET':
        return flask.render_template('index.html')

    if flask.request.method == 'POST':
        
        sequence = flask.request.form['sequence']

        # if it's shorter than 10 residues, take it as a identifier
        if re.search("[0-9]", sequence) and sequence[0] != ">":
            # recognize if it's an identifier and which and output sequence or error.
            sequence = identifier2fasta(sequence)           
        else:
            # get ride of newlines,etc and header if it's fasta  
            sequence = clean_input(sequence)
        
        try:
            predictions, fasta, csv = predict_full(sequence.replace(' ',''))
            plot = create_plot(predictions)
        except Exception as e:
            return flask.render_template("try_again.html")

    return flask.render_template("results.html", name='ADPred', plot=plot, csv_data=csv)


@app.route('/predictions/<filename>')
def download(filename):
    #uploads = os.path.join(current_app.root_path, app.config['predictions'])
    #return send_from_directory(directory=uploads, filename=filename)
    return flask.send_from_directory('predictions', filename=filename) 


# With debug=True, Flask server will auto-reload 
# when there are code changes
if __name__ == '__main__':
    app.run(port=5000, debug=True)
