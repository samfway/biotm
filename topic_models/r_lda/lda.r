# Wrapper to interface Python with R's LDA package.

# __author__ = "Sam Way"
# __copyright__ = "Copyright 2014, The Clauset Lab"
# __license__ = "BSD"
# __maintainer__ = "Sam Way"
# __email__ = "samfway@gmail.com"
# __status__ = "Development"

library('optparse')
library('lda')

args <- commandArgs(trailingOnly=TRUE)
if(!is.element('--source_dir', args) && 
   !(is.element('--help', args) || is.element('-h', args)))
    stop("\n\nPlease use '--source_dir' to specify the R source code directory.\n\n")
sourcedir <- args[which(args == '--source_dir') + 1]
source(sprintf('%s/file_io.r',sourcedir))

# Make option list and parse command line
option_list <- list(
    make_option(c("--source_dir"), type="character",
                help="Path to R source directory [required]."),
    make_option(c("-m", "--mode"), type="character",
                help="Mode ('est' or 'inf') [required]"),
    make_option(c("-i", "--datafile"), type="character",
                help="Input data file [required]."),
    make_option(c("-l", "--labelfile"), type="character",
                help="Input labels [required for est]."),
    make_option(c("-a", "--algo"), type="character",
                help="Algorithm ('lda' or 'slda')",
                default="lda"),
    make_option(c("-s", "--model"), type="character",
                help="Path to stored model"),
    make_option(c("-k", "--topics"), type="integer",
                help="Number of topics", default=10),
    make_option(c("-w", "--words"), type="integer",
                help="Number of words/size of vocabulary"),
    make_option(c("-o", "--outdir"), type="character", default='.',
                help="Output directory [default %default]")
)
opts <- parse_args(OptionParser(option_list=option_list), args=args)

# Error Checking 
if (is.null(opts$datafile)) stop('Please supply a data file')
if (is.null(opts$words)) stop('Please specify size of vocabulary')
if (is.null(opts$algo)) stop('Please supply an algorithm (lda/slda)')
if (!is.element(opts$algo, c('lda', 'slda'))) 
    stop('Please supply a valid algorithm (lda/slda)')
if (is.null(opts$mode)) stop('Please supply a mode (est/inf)')
if (is.null(opts$model)) stop('Please supply a model file to be loaded/saved')
if (opts$mode == "inf" && is.null(opts$model)) stop('Please supply a model file')
if (opts$topics < 1) stop('Number of topics must be greater than one')
if (opts$algo == "slda" &&  # If SLDA and training, must have a labels file
        opts$mode == "est" && 
        is.null(opts$labelfile)) 
    stop('Please supply a labels file')

# Create output directory if needed
if(opts$outdir != ".") dir.create(opts$outdir,showWarnings=FALSE, recursive=TRUE)
data <- ParseDataFile(opts$datafile)
vocab = unlist(strsplit(toString(1:opts$words),"\\, "))

if (opts$algo == "slda") 
{ 
    if (opts$mode == "est")
    {
        params <- sample(c(-1, 1), opts$topics, replace=TRUE)
        labels <- ParseLabelFile(opts$labelfile)
                
        res <- slda.em(documents=data$documents, 
                       K=opts$topics, 
                       vocab=vocab, 
                       num.e.iterations=100, 
                       num.m.iterations=40,
                       alpha=1.0, eta=0.1,
                       annotations=labels,
                       params=params,
                       variance=0.25)
        
        save(res, file=opts$model)
    }
    else # "inf"
    {
        attach(opts$model)
        if (!exists("res"))
            stop('Error loading trained model!')
        docsums <- slda.predict.docsums(documents = data$documents,
                                        topics = res$topics,
                                        alpha = 1.0,
                                        eta = 0.1)
        
        write.table(t(docsums), file=paste(opts$outdir, "tc.out", sep="/"),
                    col.names=FALSE, row.names=FALSE)
    }
}

if (opts$algo == "lda")
{    
    if (opts$mode == "est")
    {
        res <- lda.collapsed.gibbs.sampler(documents = data$documents, 
                                           K = opts$topics, 
                                           vocab = data$vocab,
                                           num.iterations = 100,
                                           alpha = 1.0, eta = 0.1)
        save(res, file=opts$model)
    }
    else  # "inf"
    {
        attach(opts$model)
        if (!exists("res"))
            stop('Error loading trained model!')
        inf.res <- lda.collapsed.gibbs.sampler(documents = data$documents,
                                               K = opts$topics,
                                               vocab = data$vocab,
                                               num.iterations = 100,
                                               alpha = 1.0,  eta = 0.1,
                                               initial = res$aassignments,
                                               freeze.topics = TRUE)
        
        write.table(t(inf.res$document_sums), 
                    file=paste(opts$outdir, "tc.out", sep="/"),
                    col.names=FALSE, row.names=FALSE)
    }
}
