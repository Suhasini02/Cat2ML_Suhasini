from flask import Flask
import login as lg
from flask import Flask, render_template,request
import pickle

app=Flask(__name__)
@app.route('/')
def login():
    return render_template('login.html')
with open('model1.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/auth', methods=['POST'])
def auth():
    user     = request.form['uname']
    password = request.form['psw']

    # check if the username and password are valid
    usename, pswd = lg.getcredentials()
    if user in usename:
      if password  in pswd:
         return render_template('home.html')
    else:
        # if the username and password are not valid, redirect back to the login page
        return render_template('login.html', error='Invalid username or password')
    
@app.route("/predict",methods=["POST"])
def predict():
     a = (request.form['Genre'])
     b = (request.form['Name'])
     prediction = model.predict([[a,b]])[0]
     print("krithiga")
     if(prediction):
       return render_template("result.html" , value = prediction)



if __name__ == '__main__':
    app.run(debug=True)