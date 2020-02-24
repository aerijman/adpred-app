from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():

    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        try:
            protId = request.form['protIdName']
            sequence = identifier2fasta(protId)
        except Exception as e:
            protId='you selected seequence'

        try:
            sequence = request.form['Sequence']        
        except Exception as e:
            sequence='you selected protId'

    

    return protId, sequence

if __name__ == '__main__':
    app.run(port=5000, debug=True)
