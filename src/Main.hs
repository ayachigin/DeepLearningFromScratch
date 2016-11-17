module Main where

import Prelude hiding (map, zipWith)
import Data.Array.Repa hiding ((++))
import qualified Data.Array.Repa as R
import Data.Array.Repa.Algorithms.Matrix (mmultS)
import System.Random (randomRs, mkStdGen)
import Control.Lens

import Mnist (randomNormalizedDatasets, Label, NormalizedDataSets)
import NeuralNetwork

batsize :: Int
batsize = 100

main :: IO ()
main = do
  d <- dataset
  n <- network [768, 50, 100, 10] ((-10), 10)
  r <- performNN d n batsize
  print r
  where
    dataset :: IO NormalizedDataSets
    dataset = randomNormalizedDatasets batsize
