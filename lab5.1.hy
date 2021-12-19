(import [numpy :as np])
(import [pandas :as pd])

(setv data (pd.read_csv "stat.csv"))

(setv score (get data "score"))
(setv mean (/ (np.sum score) (len score)))
(setv squares (np.sum (* score score)))
(setv dispersion (- (/ squares (len score)) (* mean mean)))
(print "Dispersion:" dispersion)

(setv time (get data "time"))
(print "Math expectation:"  (/ (np.sum time) (len time)))
