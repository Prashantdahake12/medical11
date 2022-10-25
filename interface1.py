from flask import Flask,jsonify,render_template,request,json

import config


app=Flask(__name__)   # instance creation
############################home api###########################################
@app.route('/') #home api
def hello_flask():
    print('Welcome to Flask')
    return render_template("home.html")
###########################test api##########################################



if __name__ == "__main__":
    app.run(host='0.0.0.0',port=config.PORT_NUMBER,debug=False)

