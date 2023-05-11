function [documents] = preprocessReviews(textData)
% Convert all of the text data into lowercase.
cleanTextData = lower(textData);
% Tokenize the textData.
documents = tokenizedDocument(cleanTextData);
% Erase all of the punctuation. 
documents = erasePunctuation(documents);
% Remove a list of stop words.
documents = removeStopWords(documents); end