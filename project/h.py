from flask import Flask
app=Flask(__name__)

@app.route("/")
def index():
    return("home page")
@app.route("/xyz")
def abc():
    return("home page1")



if __name__ == "__main__":
    app.run(debug=True)
