module Main where

type Weight = Double
type Threshold = Double

data Perceptron = Perceptron { w1 :: Double
                             , w2 :: Double
                             , theta :: Double
                             } deriving (Show, Read, Ord, Eq)

main :: IO ()
main = do
  putStrLn "hello world"

andP :: Num a => Double -> Double -> a
andP = perceptron (Perceptron 0.5 0.5 0.8)

perceptron :: Num a => Perceptron -> Double -> Double -> a
perceptron (Perceptron w1 w2 t) x y = if w1 * x + w2 * y <= t then
                                        0
                                      else
                                        1
