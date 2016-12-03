function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
%   produces a feature vector from the word indices. 

% Total number of words in the dictionary
vocabList = getVocabList();
n = length(vocabList)

% You need to return the following variables correctly.
x = zeros(n, 1);
L = length(word_indices)

for i = 1:L
    x(word_indices(i)) = 1;
end


end
