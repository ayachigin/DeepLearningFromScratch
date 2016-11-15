module Main where

import Test.DocTest

main :: IO ()
main = doctest ["-isrc", "src/Main3.6.hs"]
