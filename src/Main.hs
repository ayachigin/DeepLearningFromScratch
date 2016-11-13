{-# LANGUAGE DeriveFunctor #-}
module Main where

--import Numeric.LinearAlgebra

import Data.List (transpose)

data Matrix = V0 Double
            | V1 [Double]
            | V2 [[Double]]
            deriving (Show, Read, Ord, Eq)

main :: IO ()
main = do
  let x = V1 [1.0, 0.5]
      w1 = V2 [ [0.1, 0.3, 0.5]
              , [0.2, 0.4, 0.6]]
      b1 = V1 [0.1, 0.2, 0.3]
      a1 = add (dot x w1) b1
      z1 = apply sigmonoid a1
      w2 = V2 [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]
      b2 = V1 [0.1, 0.2]
      a2 = add (dot z1 w2) b2
      z2 = apply sigmonoid a2
      w3 = V2 [[0.1, 0.3], [0.2, 0.4]]
      b3 = V1 [0.1, 0.2]
      a3 = add (dot z2 w3) b3
      z3 = apply id a3
  print z3

sigmonoid x = 1 / (1 + exp (-x))

softmax x = let c = maximum x
                expA = map (\a -> exp $ a - c) x
                sumExpA = sum expA
            in map (/sumExpA) expA


dot :: Matrix -> Matrix -> Matrix
dot x y = case (x, y) of
            ((V1 xs), (V1 ys)) -> V0 $ dot' xs ys
            ((V2 xs), (V1 ys)) -> V1 $ map (\ls -> dot' ls ys) xs
            ((V1 xs), (V2 ys)) -> V1 $ dot'' xs ys
            ((V2 xs), (V2 ys)) -> V2 $ map (\x -> dot'' x ys) xs
  where
    dot' = (sum .) . zipWith (*)
    dot'' a = map (dot' a) . transpose

apply :: (Double -> Double) -> Matrix -> Matrix
apply f (V0 a) = V0 $ f a
apply f (V1 ls) = V1 $ map f ls
apply f (V2 ls) = V2 $ map (map f) ls

add :: Matrix -> Matrix -> Matrix
add (V0 x) (V0 y) = V0 $ x + y
add (V0 x) (V1 y) = V1 $ map (+x) y
add (V1 x) (V0 y) = V1 $ map (+y) x
add (V0 x) (V2 y) = V2 $ map (map (+x)) y
add (V2 x) (V0 y) = V2 $ map (map (+y)) x
add (V1 x) (V1 y) = V1 $ zipWith (+) x y
