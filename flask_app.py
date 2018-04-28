from flask import Flask, jsonify
from flask import abort
import sqlite3
import pandas as pd
import pickle
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree

app = Flask(__name__)

final_df=pd.read_csv('finaldata.csv')

user_similarity_model = pickle.load(open("user_similarity.pkl", "rb"))


@app.route('/getSimilarUsers/<int:user_id>', methods=['GET'])
def get_task(user_id):
    row=final_df.iloc[user_id-1,1:]
    dist,index = user_similarity_model.query([row.values], k=6)
    fin_index = np.delete(index[0],0)
    finout = pd.Series(fin_index).to_json(orient='values')
    if len(row) == 0:
        abort({"status":"201","message":"User doesn't exist"})
    return finout

if __name__ == '__main__':
    app.run(debug=True)