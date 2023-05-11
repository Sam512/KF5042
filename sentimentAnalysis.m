% Sentiment Analysis using Bing Liu opinion lexicon
% Code supported from the week 9 workshop

% Clear workspace
clc;
clear;

% Read positive and negative words from lexicon
[wordsPositive, wordsNegative] = readOpinionLexicon();

% Create hash table for sentiment scores
words_hash = createHashTable(wordsPositive, wordsNegative);

% Collect review data from file
filename = 'flight_tweets.txt';
[dataReviews, textData, actualScore] = collectReviewData(filename);

% Preprocess reviews
sents = preprocessReviews(textData);
fprintf('File: %s, Sentences: %d \n', filename, size(sents));

% Calculate sentiment scores
sentimentScore = calculateSentimentScores(sents, words_hash, actualScore);

% Display results and confusion matrix
displayResultsAndConfusionMatrix(sentimentScore, actualScore);

% Function to read opinion lexicon
function [wordsPositive, wordsNegative] = readOpinionLexicon()
    % Read in all of the positive words
    fidPositive = fopen(fullfile('opinion-lexicon-English', 'positive-words.txt'));
    C = textscan(fidPositive, '%s', 'CommentStyle', ';');
    wordsPositive = string(C{1});
    fclose(fidPositive);

    % Read in all of the negative words
    fidNegative = fopen(fullfile('opinion-lexicon-English', 'negative-words.txt'));
    C = textscan(fidNegative, '%s', 'CommentStyle', ';');
    wordsNegative = string(C{1});
    fclose(fidNegative);
end

% Function to create hash table for sentiment scores
function words_hash = createHashTable(wordsPositive, wordsNegative)
    words_hash = java.util.Hashtable;
    [possize, ~] = size(wordsPositive);
    for ii = 1:possize
        words_hash.put(wordsPositive(ii, 1), 1);
    end
    [negsize, ~] = size(wordsNegative);
    for ii = 1:negsize
        words_hash.put(wordsNegative(ii, 1), -1);
    end
end

% Function to collect review data from file
function [dataReviews, textData, actualScore] = collectReviewData(filename)
    dataReviews = readtable(filename, 'TextType', 'string');
    textData = dataReviews.review;
    actualScore = dataReviews.score;
end

% Function to calculate sentiment scores
function sentimentScore = calculateSentimentScores(sents, words_hash, actualScore)
    sentimentScore = zeros(size(sents));

    for ii = 1:sents.length
        docwords = sents(ii).Vocabulary;
        for jj = 1:length(docwords)
            if words_hash.containsKey(docwords(jj))
                sentimentScore(ii) = sentimentScore(ii) + words_hash.get(docwords(jj));
            end
        end

        if (sentimentScore(ii) >= 1)
            sentimentScore(ii) = 1;
        elseif (sentimentScore(ii) <= -1)
            sentimentScore(ii) = -1;
        end
        fprintf('Sent: %d, words: %s, FoundScore: %d, GoldScore: %d\n', ...
            ii, joinWords(sents(ii)), sentimentScore(ii), actualScore(ii));
    end
end

% Function to display results and confusion matrix
function displayResultsAndConfusionMatrix(sentimentScore, actualScore)
    ZeroVal = sum(sentimentScore == 0);
    covered = numel(sentimentScore) - ZeroVal;
    fprintf("Total of positive and negative classes (coverage): %2.2f%%, Distinct %d, Not Found or Neutral: %d\n", ...
        (covered * 100) / numel(sentimentScore), covered, ZeroVal);

    % Calculate true positive
    tp = sentimentScore((sentimentScore == 1) & (actualScore == 1));
    % Calculate false positive
    fp = sentimentScore((sentimentScore == 1) & (actualScore == 0));
    % Calculate true negative
    tn = sentimentScore((sentimentScore == -1) & (actualScore == 0));
    % Calculate false negative
    fn = sentimentScore((sentimentScore == -1) & (actualScore == 1));

    % Calculate accuracy, precision, recall, and F1 score
    acc = (numel(tp) + numel(tn)) * 100 / covered;
    precision = numel(tp) / (numel(tp) + numel(fp));
    recall = numel(tp) / (numel(tp) + numel(fn));
    f1 = 2 * precision * recall / (precision + recall);

    % Display evaluation metrics
    fprintf("Accuracy: %2.2f%%, Precision: %2.2f%%, Recall: %2.2f%%, F1 score: %2.2f%%\n", acc, precision*100, recall*100, f1*100);

    % Generate confusion matrix of the results
    confusion = confusionchart(actualScore, sentimentScore);
    confusion.FontSize = 12;
    confusion.Normalization = 'total-normalized';
end

