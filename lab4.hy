(import [numpy :as np]
        [matplotlib [pyplot :as plt]]
        [seaborn :as sns]
        [pandas :as pd]
        [sklearn.preprocessing [OrdinalEncoder]]
        [sklearn.model_selection [train_test_split]]
        [sklearn.linear_model [LinearRegression]]
)

(defn split_data [data]
    [(data.drop "score" :axis 1) data.score])


(setv data (pd.read_csv "stat.csv" ))
(print (data.head))

(setv test_data
(cut data 0 6))

(setv train_data (cut data 6 (len data)))



(setv x (get (split_data train_data) 0))
(setv y (get (split_data train_data) 1))

(setv split (train_test_split x y :test_size 0.2 :shuffle True))
(setv x_train (get split 0))
(setv x_valid (get split 1))
(setv y_train (get split 2))
(setv y_valid (get split 3))

(setv cat_cols ["win_or_lose" "alg"])
(setv encoder (OrdinalEncoder))
(assoc x_train cat_cols (encoder.fit_transform (get x_train cat_cols)))
(assoc x_valid cat_cols (encoder.transform (get x_valid cat_cols)))

(setv model (LinearRegression))
(model.fit x_train y_train)
(print model.coef_)
(print (+ "R2 score:" (str (model.score x_valid  y_valid))))


(setv x_test (get (split_data test_data) 0))
(setv y_test (get (split_data test_data) 1))
(assoc x_test cat_cols (encoder.transform (get x_test cat_cols)))
(setv y_pred (model.predict x_test))


(setv stats (pd.DataFrame {
    "prediction" (np.abs (- y_test y_pred))
    "real" (np.abs(- y_test (y_train.mean)))
 }))



(print stats)