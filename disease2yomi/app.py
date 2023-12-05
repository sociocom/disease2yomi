from flask import Flask, render_template, request
from estimate import estimate_yomi, estimate_icd10


app = Flask(__name__)
PREFIX = "/diseasetoyomi"


@app.route(PREFIX + "/")
def index():
    name = request.args.get("name")
    return render_template("index.html")


@app.route(PREFIX + "/result", methods=["post"])
def post():
    disease_name = request.form["name"]
    yomi = estimate_yomi(disease_name)
    icd10 = estimate_icd10(disease_name)
    return render_template(
        "result.html",
        disease_name=disease_name,
        yomi=yomi,
        icd10=icd10,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=10101)
    # app.run(debug=True, port=5001)
