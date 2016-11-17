module Main where

import Prelude hiding (map, zipWith)
import Data.Array.Repa hiding ((++))
import qualified Data.Array.Repa as R
import Data.Array.Repa.Algorithms.Matrix (mmultS)
import Data.Array.Repa.Repr.Vector (V, computeVectorP, fromListVector)
import System.Random (randomRs, mkStdGen)
import Control.Lens

import Mnist (choiceIO, toArray, normalize, Label)
import NeuralNetwork

main :: IO ()
main = do
  d <- dataset
  as <- computeVectorP $ map (performNNStep network) d
  print $ (/(fromIntegral batsize)) . sum . toList $ as
  where
    dataset :: IO (Array V DIM1 (Array U DIM2 Double, Label))
    dataset = do
      datasets <- fmap (normalize) $ choiceIO batsize
      return . toArray batsize $ datasets
    performNNStep :: NN -> (Matrix U, Int) -> Double
    performNNStep n (d, l) =
      let z = predict d n
      in calcLoss crossEntropyError z l
    network = [ (x1, b1, sigmonoid)
              , (x2, b2, sigmonoid)
              , (x3, b3, softmax) ]
    s1 = 28 * 28 * 50
    x1 = matrix (ix2 784 50) s1 ((-10), 10.0) 10
    b1 = matrix (ix2 1 50) 50 (0.0, 1.0) 10
    x2 = matrix (ix2 50 100) (50*100) ((-10), 100) 30
    b2 = matrix (ix2 1 100) 100 ((-10), 100) 20
    x3 = matrix (ix2 100 10) (100 * 10) ((-10), 100) 5
    b3 = matrix (ix2 1 10) 10 ((-10), 10) 100
