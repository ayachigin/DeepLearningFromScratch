module Main where

import System.IO (stdout, hFlush)
import Control.Monad (foldM_)
import Data.Array.Repa hiding (map, zipWith, (++))
import qualified Data.Array.Repa as R
import Data.Array.Repa.Algorithms.Matrix (row, col, mmultS)

import Mnist (randomNormalizedDataset, NormalizedDataSet)
import NeuralNetwork

import Util

batsize :: Int
batsize = 30

learningRate :: Double
learningRate = 0.1

iterNum :: Int
iterNum = 20

main :: IO ()
main = do
  let x = delay $ fromListUnboxed (ix2 2 2) [0.6, 0.9, 0.2, 0.3]
      y = fromListUnboxed (ix2 2 2) [0, 1, 1, 0]
  p $ crossEntropyError (softmax x) y 2
  d <- dataset
  n <- network [28*28, 20, 10] ((-1.0), 1.0)
  putStrLn . ("Accuracy:"++) . format $ accuracy d batsize n
  putStrLn . ("Loss    :"++) . (show)  $ loss d batsize n
  foldM_ (\ni _ -> performNN ni) n [1..iterNum]
  where
    performNN n = do
      d <- dataset
      g <- numericalGradient d batsize n
      n' <- gradientDescent learningRate n g
      putStrLn . ("Accuracy:"++) . format $ accuracy d batsize n
      putStrLn . ("Loss    :"++) . (show)  $ loss d batsize n
      return n'
    g (w, b) = (w, b, sigmonoid)
    k ls = updateL ls (length ls-1) (\(w, b, _) -> (w, b, softmax))
    f (w, b, _) = (w, b)
    p s = print s >> hFlush stdout
    format = (++"%"). show . floor . (*100)
    dataset :: IO NormalizedDataSet
    dataset = randomNormalizedDataset batsize
