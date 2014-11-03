# Functions for interfacing with LDA files

# __author__ = "Sam Way"
# __copyright__ = "Copyright 2014, The Clauset Lab"
# __license__ = "BSD"
# __maintainer__ = "Sam Way"
# __email__ = "samfway@gmail.com"
# __status__ = "Development"

"ParseDataFile" <- function(filename)
{
    input.fp <- file(filename, open="r")
    docs <- c()
    max.word <- 0
    max.count <- 0
    
    while (length(line <- readLines(input.fp, n=1, warn=FALSE)) > 0)
    {
        pieces <- strsplit(line, " ")[[1]]
        if (length(pieces) < 2) 
        {
            if (length(pieces) == 1)
            {
                if (as.integer(line) == 0)
                    stop("Document with no words detected!")
            }
            next  # skip empty lines
        }
        
        num.pieces = as.integer(pieces[1])
        if (num.pieces+1 != length(pieces))
            stop("Improperly formatted data file[1]!")
        
        words <- vector("integer", num.pieces)
        counts <- vector("integer", num.pieces)
        for (i in 1:num.pieces) 
        {
            pair <- strsplit(pieces[i+1], ":")[[1]]
            if (length(pair) != 2)
                stop("Improperly formatted data file[2]!")
            words[i] <- as.integer(pair[1])
            counts[i] <- as.integer(pair[2])
            if (words[i] > max.word)
                max.word <- words[i]
            if (counts[i] > max.count)
                max.count <- counts[i]
        }
        
        doc <- rbind(words, counts)
        docs[[length(docs)+1]] <- doc
    }
    
    close(input.fp)
    vocab = unlist(strsplit(toString(1:(max.word+1)),"\\, "))
    
    print(c("Max word count is...", max.count))
    
    return(list("documents"=docs, "vocab"=vocab))
}

"ParseLabelFile" <- function(filename)
{
    input.fp <- file(filename, open="r")
    labels <- as.numeric(readLines(input.fp, warn=FALSE))
    close(input.fp)
    return(labels)
}