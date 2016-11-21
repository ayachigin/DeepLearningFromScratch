module Main where

import System.IO (stdout, hFlush)
import Control.Monad (foldM_)
import Data.Array.Repa (fromListUnboxed, ix2)

import Mnist (randomNormalizedDataset, NormalizedDataSet)
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
  let y = fromListUnboxed (ix2 2 2) [0.6, 0.9, 0.2, 0.3]
  let t = fromListUnboxed (ix2 2 2) [0, 1, 1, 0]
  let e = crossEntropyError y t 2

  print $ e =~ 0.857399214046 $ 3
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
