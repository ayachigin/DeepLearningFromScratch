{-# LANGUAGE DeriveFunctor #-}
module Main where

import Prelude hiding (map, zipWith)
import Data.Array.Repa
import Data.Array.Repa.Algorithms.Matrix

ain = do
  let w1 = (fromListUnboxed (ix2 1 2) [1.0, 0.5]::Array U DIM2 Double)
  v <- mmultP w1 x2
  print $ toList (w1:: Array U DIM2 Double)
  print $ toList v --(v :: Array U DIM2 Double)
  return ()
      --where

x2 :: Array U DIM2 Double
x2 = fromListUnboxed (ix2 2 3) [0.1, 0.3, 0.5, 0.2, 0.4, 0.6]
xs' :: Array U DIM1 Int
xs' = fromListUnboxed (ix1 2) [1, 5]

type Matrix = Array U DIM2 Double

main :: IO ()
main = do
  let x  = fromListUnboxed (ix2 1 2) [1.0, 0.5]
      w1 = fromListUnboxed (ix2 2 3) [0.1, 0.3, 0.5, 0.2, 0.4, 0.6]
      b1 = fromListUnboxed (ix2 1 3) [0.1, 0.2, 0.3]
      a1 = zipWith (+) b1 $ mmultS x w1
      z1 = computeS $ map sigmonoid a1
      w2 = fromListUnboxed (ix2 3 2) [0.1, 0.4, 0.2, 0.5, 0.3, 0.6]
      b2 = fromListUnboxed (ix2 1 2) [0.1, 0.2]
      a2 = zipWith (+) b2 $ mmultS z1 w2
      z2 = computeS $ map sigmonoid a2
      w3 = fromListUnboxed (ix2 2 2) [0.1, 0.3, 0.2, 0.4]
      b3 = fromListUnboxed (ix2 1 2) [0.1, 0.2]
      a3 = zipWith (+) b3 $ mmultS z2 w3
      z3 = a3
  print $ toList z3

sigmonoid :: Double -> Double
sigmonoid x = 1 / (1 + exp (-x))

softmax :: [Double] -> [Double]
softmax x = let c = maximum x
                expA = fmap (\a -> exp $ a - c) x
                sumExpA = sum expA
            in fmap (/sumExpA) expA
