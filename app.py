from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model     = joblib.load('./loan/model.pkl')
encoders  = joblib.load('./loan/encoders.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':

        data_baru = {
            'person_age':        int(request.form['age']),
            'person_gender':     request.form['gender'].lower(),
            'person_education': request.form['education'],
            'person_income':     float(request.form['income']),
            'person_emp_exp':int(request.form['experience']),
            'person_home_ownership':  request.form['ownership'],
            'loan_amnt':   float(request.form['loanamnt']),
            'loan_intent' : request.form['loanintent'],
            'loan_int_rate' :float(request.form['loanrate']),
            'loan_percent_income' :float(request.form['loaninc']),
            'cb_person_cred_hist_length' :float(request.form['cb']),
            'credit_score': int(request.form['cs']),
            'previous_loan_defaults_on_file' : request.form['prevl']
        }
    
        df = pd.DataFrame([data_baru])


        for col, le in encoders.items():
            df[col] = le.transform(df[col])

        pred = model.predict(df)[0]  

        prediction = f"Status pinjaman diprediksi: {"Disetujui" if pred == 1 else "Tidak Disetujui"}"


    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
