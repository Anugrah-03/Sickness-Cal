from flask import Flask,render_template,request
import pickle
import numpy as np
from flask_ngrok import run_with_ngrok
app=Flask(__name__)
model=pickle.load(open('diabetes.pkl','rb'))
model1=pickle.load(open('Alzehimer.pkl','rb'))
model2=pickle.load(open('heart_failure.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route("/about.html")
def about():
    return render_template("about.html")
@app.route("/contact.html")
def contact():
    return render_template("contact.html")
@app.route('/selection_page.html')
def selectionpage():
    return render_template('selection_page.html')
@app.route('/diabetes_page.html')
def diabetes_page():
    return render_template('diabetes_page.html')
@app.route('/result_diabetes',methods=['POST'])
def predict():
    data1=float(request.form['a'])
    data1=float(request.form['a'])
    data2=float(request.form['b'])
    data3=float(request.form['c'])
    data4=float(request.form['d'])
    data5=float(request.form['e'])
    data6=float(request.form['f'])
    data7=float(request.form['g'])
    data8=float(request.form['h'])
    arr=np.array([[data8,data1,data2,data3,data4,data5,data6,data7]]).astype('float')
    pred=model.predict(arr)
    print(pred)
    return render_template('result_diabetes.html',prediction_text=pred)
@app.route('/Alzheimers_page.html')
def Alzehimer_page():
    return render_template('Alzheimers_page.html')
@app.route('/result_Alzheimer',methods=['POST'])
def pred():
    data1=request.form['a']
    data2=float(request.form['b'])
    data3=float(request.form['c'])
    data4=float(request.form['d'])
    data5=float(request.form['e'])
    data6=float(request.form['f'])
    data7=float(request.form['g'])
    data8=float(request.form['h'])
    if(data1.capitalize()=="MALE"):
        data1=1
    else:
        data1=0
    print(data1)
    arr=np.array([[data1,data2,data3,data4,data5,data6,data7,data8]]).astype('float')
    pred=model1.predict(arr)
    print(pred)
    return render_template('result_Alzheimers.html',prediction_text=pred)
@app.route('/heartfailure_page.html')
def heartfailure_page():
    return render_template("heartfailure_page.html")
@app.route('/result_heartfailure',methods=['POST'])
def predi():
    data1=float(request.form['g1'])
    data2=float(request.form['b'])
    data3=float(request.form['c'])
    data4=float(request.form['d'])
    data5=float(request.form['e'])
    data6=float(request.form['f'])
    data7=float(request.form['g'])
    data8=float(request.form['h'])
    data9=float(request.form['i'])
    data10=request.form['j']
    data11=request.form['k']
    data12=float(request.form['l'])
    if(data10.capitalize()=="MALE"):
        data10=1
    else:
        data10=0
    if(data11.capitalize()=="Yes"):
        data11=1
    else:
        data11=0 
    arr=np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12]]).astype('float')
    pred=model2.predict(arr)
    print(pred)
    return render_template('result_heartfailure.html',prediction_text=pred)
    

if __name__=='__main__':
    app.run()