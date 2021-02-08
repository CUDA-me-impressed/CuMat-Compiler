; ModuleID = 'CuMat-testOne.cm'
source_filename = "CuMat-testOne.cm"

%matHeader = type { [1 x i64]*, i64, i64 }

define %matHeader* @main() {
main_entry:
  %malloccall = tail call i8* @malloc(i32 ptrtoint (i64* getelementptr (i64, i64* null, i32 1) to i32))
  %matArrData = bitcast i8* %malloccall to [1 x i64]*
  %matArrPtr = getelementptr [1 x i64], [1 x i64]* %matArrData, i32 0
  %malloccall1 = tail call i8* @malloc(i32 ptrtoint (%matHeader* getelementptr (%matHeader, %matHeader* null, i32 1) to i32))
  %matStruct = bitcast i8* %malloccall1 to %matHeader*
  %0 = getelementptr inbounds %matHeader, %matHeader* %matStruct, i32 0, i32 0
  store [1 x i64]* %matArrPtr, [1 x i64]** %0
  %1 = getelementptr inbounds %matHeader, %matHeader* %matStruct, i32 0, i32 1
  store i64 0, i64* %1
  %2 = getelementptr inbounds %matHeader, %matHeader* %matStruct, i32 0, i32 2
  store i64 8, i64* %2
  %3 = getelementptr inbounds %matHeader, %matHeader* %matStruct, i32 0, i32 0
  %matArrPtr2 = load [1 x i64]*, [1 x i64]** %3
  %4 = getelementptr inbounds [1 x i64], [1 x i64]* %matArrPtr2, i32 0, i32 0
  store i64 1, i64* %4
  ret %matHeader* %matStruct
}

declare noalias i8* @malloc(i32)

!nvvm.annotations = !{!0, !1}

!0 = !{!"kernel"}
!1 = !{%matHeader* ()* @main, !"kernel", i64 1}
