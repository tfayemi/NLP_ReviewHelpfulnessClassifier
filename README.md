# NLP_ReviewHelpfulnessClassifier

*Please veiw full ![report](https://github.com/tfayemi/NLP_ReviewHelpfulnessClassifier/blob/main/report.pdf) for complete breakdown of the models, how they work, and the intended results.*

A collection of machine learning models trained on the Amazon's 2018 Review Data to predict wether a product review (free text) will be helpful or unhelpful for other potential buyers.

How To Use: 
1. Download the ‘helpful’ package and store it wherever you’d like. 

2. From your terminal or cmd line, run
$ python3 helpful_api.py

3. In another terminal/cmd line window, send you API requests in the following format (I didn’t have time to make a GUI unfortunately!)

$ curl -X POST -F "review=your review here" http://127.0.0.1:5000/predict

Replace “your review here” with your review!

And you should receive a JSON response to your terminal that looks something like this:

![Example](https://raw.githubusercontent.com/tfayemi/NLP_ReviewHelpfulnessClassifier/main/images/helpful_example.png)

HOPE IT WORKS! :)
