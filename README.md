## Machine Learning for Exo-Planet detection

This repository contains code developed for MSc research project at Warwick University [Mathematics of Systems CDT](http://www2.warwick.ac.uk/fac/sci/mathsys/) under the title 
of "Machine Learning for Exoplanet Detection". (**On going project**)

##### Supervised by:

Dr. David Armstrong (Astrophysics, Warwick University)
Dr. Theo Damoulas   (Computer Science, Warwick University)
    
##### Background:

When searching for new planets through transit detection in NASA's Kepler satellite data, a significant portion of time is spent on validation of the detected signal. For example, light curve fitting of the possible false positive scenarios, and follow-up observations using other methodologies. Using pre-trained models would allow researchers to quickly validate planets and investigate only uncertain signals, particularly in future, data abundant, missions such as PLATO. We optimize and evaluate the performance of multiple machine learning techniques in classifying planet candidates into false positive and confirmed planets, using publicly available Kepler light curve data. The data is transformed into real-numbered attributes through Kepler science pipeline and additionally computed attributes, such as ephemeris correlation and statistic derived from self-organized maps. First, we test the classification accuracy and provide interpretation to the results obtained. Second, we evaluate the quality of probabilities obtained from machine learning models against those acquired from MCMC based planet validation method - **vespa**. Identifying a significant disagreement between probabilities for yet not dispositioned planets in Kepler data, we test the latter for novelty. To optimize the interpretability of machine learning for planet validation we test sparse Gaussian process classification, which has the advantage of scalability and posterior variance in prediction. High posterior probability variance was not achieved for false positive instances where machine learning predicts high false positive probability contrary to \textbf{vespa}, likely because of some of the false positive scenarious appearing in disposition data not accounted for by the MCMC based model. The converse appears to be the case for not dispositioned data, where **vespa** provides confident probabilities and machine learning methods are less certain. Signaling a possible difference in false positive cases composition in labeled and unlabeled data. Most of machine learning models capture 99% of false positives and 74% to 92% of confirmed planets, with Gaussian process based model identifying misclassified planets as high uncertainty predictions, pointing out instances where researchers input is needed.

##### Direct Link to tables referenced in footnotes:

* [Footnote 4 - Full CSV table of parameter search on the training data](https://github.com/jgamper/MScProject-Kepler-ML/blob/master/data/output/classifiersRun_M8D23H18M12/summary_M8D23H18M12.csv)

* [Footnote 5 - Full CSV results table for *uncalibrated* models](https://github.com/jgamper/MScProject-Kepler-ML/blob/master/data/output/classifiersRun_M8D23H18M12/not_calibed_df.csv)

* [Footnote 5 - Full CSV results table for *calibrated* models](https://github.com/jgamper/MScProject-Kepler-ML/blob/master/data/output/classifiersRun_M8D23H18M12/calibed_df.csv)
    

##### Link to Kepler data:

[Dataset link](http://exoplanetarchive.ipac.caltech.edu/docs/data.html)
