from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            LIMIT_BAL=int(request.form.get('LIMIT_BAL')),
            AGE=int(request.form.get('AGE')),
            BILL_AMT1=int(request.form.get('BILL_AMT1')),
            PAY_AMT1=int(request.form.get('PAY_AMT1')),
            PAY_AMT2=int(request.form.get('PAY_AMT2')),
            PAY_AMT3=int(request.form.get('PAY_AMT3')),
            PAY_AMT4=int(request.form.get('PAY_AMT4')),
            PAY_AMT5=int(request.form.get('PAY_AMT5')),
            PAY_AMT6=int(request.form.get('PAY_AMT6')),
            SEX = request.form.get('SEX'),
            EDUCATION= request.form.get('EDUCATION'),
            MARRIAGE = request.form.get('MARRIAGE'),
            PAY_0 = request.form.get('PAY_0'),
            PAY_2= request.form.get('PAY_2'),
            PAY_4 = request.form.get('PAY_4'),
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)