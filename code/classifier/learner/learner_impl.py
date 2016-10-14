# !/usr/bin/env python

import sklearn

class Learner(object):

    def __init__(self, option):
        """Initialize learner parameters

        """
          
        self.option = option

    
    def _parse_option(self, option):
        """Parse options for learner object

        """

        if not isinstance(option, str):
            raise TypeError("option parameters type is {}, but its type must be string type".format(option))
        
        options = option.strip("\r\n").split()
        i = 0
        while i < len(options):
            if i + 1 > len(options):
                raise ValueError("The options {} can not be the past parameters. ".format(options[i]))
            if options[i] == "-learner":
                if options[i+1] == "LogisticRegression":
                    clf = linear_model.LogisticRegression()
                elif options[i+1] == "LinearSVC":
                    clf = svm.LinearSVC()
                elif options[i+1] == "SVC":
                    clf = svm.SVC()
    

        

        
