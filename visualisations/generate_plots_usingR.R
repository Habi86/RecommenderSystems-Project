# Random Baseline Artist
# Fold: 2
  #1, MAP: 4.25, MAR  11.69
  #2, MAP: 4.12, MAR  15.10
  #3, MAP: 3.82, MAR  20.07
  #4, MAP: 3.70, MAR  25.02
  #5, MAP: 3.57, MAR  29.56
  #6, MAP: 3.44, MAR  33.20
  #7, MAP: 3.37, MAR  37.30
  #8, MAP: 3.26, MAR  39.40
  #9, MAP: 3.14, MAR  43.03
  #10, MAP: 3.09, MAR  45.13

# Fold: 10
#1, MAP: 0.34, MAR: 0.01
#2, MAP: 0.39, MAR: 0.02
#3, MAP: 0.33, MAR: 0.03
#4, MAP: 0.35, MAR: 0.04
#5, MAP: 0.33, MAR: 0.05
#6, MAP: 0.32, MAR: 0.06
#7, MAP: 0.32, MAR: 0.07
#8, MAP: 0.29, MAR: 0.07
#9, MAP: 0.30, MAR: 0.10
#10, MAP: 0.33, MAR: 0.10

x <- c(1:10)
#rb_MAP <- c(4.25, 4.12, 3.82, 3.70, 3.57, 3.44, 3.37, 3.26, 3.14, 3.09)
#rb_MAR <- c(11.69, 15.10, 20.07, 25.02, 29.56, 33.20, 37.30, 39.40, 43.03, 45.13)
rb_MAP <- c(0.34, 0.39, 0.33, 0.35, 0.33, 0.32, 0.32, 0.29, 0.30, 0.33)
rb_MAR <- c(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.07, 0.10, 0.10)

plot(x, rb_MAP, type="b", xaxt="n", yaxt="n", xlab="predicted artists...?", ylab="[%]", col="red", ylim=c(0,50), main="Random Baseline (random artist)")
par(new=TRUE)
plot(x, rb_MAR, type="b", xaxt="n", yaxt="n", xlab="", ylab="", col="blue", ylim=c(0,0.5))
par(new=TRUE)
axis(1, at=x, labels=x)
axis(2, at=seq(0,0.5,by=0.1))
grid()
legend(2, 0.4, c("MAP", "MAR"), lty=c(1,1), lwd=1, col=c("red", "blue"))
par(new=FALSE)



# Random Baseline User
# Fold: 2
  #1, MAP: 1.41, MAR: 0.01
  #2, MAP: 1.93, MAR: 0.02
  #3, MAP: 1.62, MAR: 0.04
  #4, MAP: 1.65, MAR: 0.04
  #5, MAP: 1.61, MAR: 0.06
  #6, MAP: 1.45, MAR: 0.07
  #7, MAP: 1.53, MAR: 0.07
  #8, MAP: 1.65, MAR: 0.08
  #9, MAP: 1.56, MAR: 0.09
  #10, MAP: 1.56, MAR: 0.10

# Fold: 10
  #1, MAP: 0.88, MAR: 11.20
  #2, MAP: 0.88, MAR: 14.77
  #3, MAP: 0.84, MAR: 19.70
  #4, MAP: 0.80, MAR: 24.20
  #5, MAP: 0.76, MAR: 28.40
  #6, MAP: 0.73, MAR: 32.20
  #7, MAP: 0.71, MAR: 35.48
  #8, MAP: 0.69, MAR: 38.67
  #9, MAP: 0.67, MAR: 41.15
  #10, MAP: 0.65, MAR: 43.62


x2 <- c(1:10)
#rb_MAP2 <- c(1.41, 1.93, 1.62, 1.65, 1.61, 1.45, 1.53, 1.65, 1.56, 1.56)
#rb_MAR2 <- c(0.01, 0.02, 0.04, 0.04, 0.06, 0.07, 0.07, 0.08, 0.09, 0.10)
rb_MAP2 <- c(0.88, 0.88, 0.84, 0.80, 0.76, 0.73, 0.71, 0.69, 0.67, 0.65)
rb_MAR2 <- c(11.20, 14.77, 19.70, 24.20, 28.40, 32.20, 35.48, 38.67, 41.15, 43.62)

plot(x2, rb_MAP2, type="b", xaxt="n", yaxt="n", xlab="predicted artists...?", ylab="[%]", col="red", ylim=c(0,50), main="Random Baseline (random artist of random user)")
par(new=TRUE)
plot(x2, rb_MAR2, type="b", xaxt="n", yaxt="n", xlab="", ylab="", col="blue", ylim=c(0,50))
par(new=TRUE)
axis(1, at=x2, labels=x2)
axis(2, at=seq(0,50,by=5))
grid()
legend(2, 40, c("MAP", "MAR"), lty=c(1,1), lwd=1, col=c("red", "blue"))
par(new=FALSE)





