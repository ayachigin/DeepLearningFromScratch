module Main where

import System.IO (stdout, hFlush)
import Control.Monad (foldM_)
import Data.Array.Repa hiding (map, zipWith)
import qualified Data.Array.Repa as R
import Data.Array.Repa.Algorithms.Matrix (row, col, mmultS)

import Mnist (randomNormalizedDataset, NormalizedDataSet)
import NeuralNetwork

import Util

batsize :: Int
batsize = 100

learningRate :: Double
learningRate = 0.1

iterNum :: Int
iterNum = 500

main :: IO ()
main = do
  d <- dataset
  n <- network [28*28, 100, 10] ((-1.0), 1.0)
  print $ accuracy d batsize n
  print $ loss d batsize n
  where
    readNN :: IO NN
    readNN = do
      n <- unpickle "nn"
      return . k . fmap g $ n
    g (w, b) = (w, b, sigmonoid)
    k ls = updateL ls (length ls-1) (\(w, b, _) -> (w, b, softmax))
    f (w, b, _) = (w, b)
    p s = print s >> hFlush stdout
    dataset :: IO NormalizedDataSet
    dataset = randomNormalizedDataset batsize
