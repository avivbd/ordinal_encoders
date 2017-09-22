# ordinal_encoders
Some sklearn style transformers which take categorical data ("foo", "bar") and encode it into ordinal data (0,1). 

LabelEncoder.py/ipynb contain sklearn\`s LabelEncoder which I modified to issue a warning rather than an error if a value not previously encountered during the fit is encountered during the transform or inverse_transform method calls. The unkown labels are contained in an array that can be accessed by calling a unknown_classes method. The encoder also allows to define custom nan values which are retained (e.g. -999) and not encoded. 

ordinal_encoder_mwe.ipynb contains a more fancy transformer which allows for imputation of values to replace any unkown values encountered during testing (the other transformer simply drops them). If unkown values are encountered during transform stage (for instance when tranforming test data), then the transformer imputes values from the categories learned during the fit. The imputation can use the most common value (mode), choose from previously encountered values in proportion to their preponderance in the data, or replace them with None. Nan values can be retained or replaced as well. 
