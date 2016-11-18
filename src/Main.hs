module Main where

import System.IO (stdout, hFlush)
import Control.Monad (foldM_)

import Mnist (randomNormalizedDatasets, NormalizedDataSets)
import NeuralNetwork

import Util

batsize :: Int
batsize = 200

learningRate :: Double
learningRate = 0.1

iterNum :: Int
iterNum = 500

main :: IO ()
main = do
  n <- readNN--network [100, 50, 10] ((-1), 1)
  foldM_(\m _ -> learnStep m) n [1..iterNum]
  print "Done"
  where
    readNN :: IO NN
    readNN = do
      n <- unpickle "nn"
      return . k . fmap g $ n
    g (w, b) = (w, b, sigmonoid)
    k ls = updateL ls (length ls-1) (\(w, b, _) -> (w, b, softmax))
    f (w, b, _) = (w, b)
    learnStep :: NN -> IO NN
    learnStep n = do
      d <- dataset
      g <- numericalGradient d n batsize
      p "gradient"
      n' <- gradientDescent learningRate n g
      p "apply gradient"
      pickle (fmap f n') "nn"
      r <- performNN d batsize n'
      p r
      return n'
    p s = print s >> hFlush stdout
    dataset :: IO NormalizedDataSets
    dataset = randomNormalizedDatasets batsize
