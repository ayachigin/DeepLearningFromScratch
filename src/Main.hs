{-# LANGUAGE DeriveFunctor #-}
module Main where

import Prelude hiding (map, zipWith)
import Data.Array.Repa
import Data.Array.Repa.Algorithms.Matrix
import Data.Foldable (foldlM)

type Matrix = Array U DIM2 Double

type Weight = Matrix

type Bias = Matrix

type ActivationFunction = Double -> Double

type Layer = (Weight, Bias, ActivationFunction)

type NN = [Layer]

main :: IO ()
main = do
  let input  = fromListUnboxed (ix2 1 2) [1.0, 0.5]
  z3 <- perfoNN input nn
  print $ toList z3

perfoNN :: Monad m => Matrix -> NN -> m Matrix
perfoNN input n = foldlM f input n
  where
    f :: Monad m => Matrix -> Layer -> m Matrix
    f i (w, b, a) = computeP . map a .
                    zipWith (+) b $ mmultS i w

nn :: NN
nn = [ ( fromListUnboxed (ix2 2 3) [0.1, 0.3, 0.5, 0.2, 0.4, 0.6]
       , fromListUnboxed (ix2 1 3) [0.1, 0.2, 0.3]
       , sigmonoid )
     , ( fromListUnboxed (ix2 3 2) [0.1, 0.4, 0.2, 0.5, 0.3, 0.6]
       , fromListUnboxed (ix2 1 2) [0.1, 0.2]
       , sigmonoid )
     , ( fromListUnboxed (ix2 2 2) [0.1, 0.3, 0.2, 0.4]
       , fromListUnboxed (ix2 1 2) [0.1, 0.2]
       , id)
     ]

sigmonoid :: Double -> Double
sigmonoid x = 1 / (1 + exp (-x))

softmax :: [Double] -> [Double]
softmax x = let c = maximum x
                expA = fmap (\a -> exp $ a - c) x
                sumExpA = sum expA
            in fmap (/sumExpA) expA
