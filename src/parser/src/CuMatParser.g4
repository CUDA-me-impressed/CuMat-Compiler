parser grammar CuMatParser;

options {
    tokenVocab = CuMatLexer;
}

program                     : EOL* imports definitions EOF ;

imports                     : cmimport* ;
cmimport                    : IMPORT_OPEN path IMPORT_CLOSE ;
path                        : (directorylist DIRSEP)? file ;
directorylist               : (directories+=directory DIRSEP)* ;
directory                   : SLUG ;
file                        : SLUG ;

definitions                 : definition* ;
definition                  : funcdef | cmtypedef | assignment ;

funcdef                     : FUNC signature EOL? block EOL ;
signature                   : typespec funcname (LPAR arguments RPAR)? ;
arguments                   : (argument (COMMA argument)* )? ;
argument                    : typespec varname ;
typespec                    : cmtypename dimensionspec? ;
dimensionspec               : LSQB dimension (COMMA dimension)* RSQB ;
dimension                   : INT | STAR ;

block                       : LBRA EOL? (assignments+=assignment EOL)* assignments+=assignment? EOL? RBRA ;
assignment                  : varname ASSIGN expression ;

expression                  : exp_logic ; //| lambda ; TODO BE ADDED
exp_logic                   : exp_comp (op_logic exp_comp)* ;
exp_comp                    : exp_bit (op_comp exp_bit)* ;
exp_bit                     : exp_sum (op_bit exp_sum)* ;
exp_sum                     : exp_mult (op_sum exp_mult)* ;
exp_mult                    : exp_pow (op_mult exp_pow)* ;
exp_pow                     : (exp_mat op_pow)* exp_mat ; // rtol
exp_mat                     : exp_neg (op_mat exp_neg)* ;
exp_neg                     : op_neg* exp_not ;
exp_not                     : op_not* exp_chain ;
exp_chain                   : (exp_func op_chain)* exp_func ;
exp_func                    : funcname args | value ;

op_logic                    : EOL? op=(LOR | LAND) EOL? ;
op_comp                     : EOL? op=(LT | GT | LTE | GTE | EQ | NEQ) EOL? ;
op_bit                      : EOL? op=(BAND | BOR ) EOL? ;
op_sum                      : EOL? op=(PLUS | MINUS) EOL? ;
op_mult                     : EOL? op=(TIMES | STAR | DIV ) EOL? ;
op_pow                      : EOL? (POW) EOL? ;
op_mat                      : EOL? (MATM) EOL? ; // Dot product symbol??
op_neg                      : EOL? (MINUS) ;
op_not                      : EOL? (LNOT | BNOT) ;
op_chain                    : EOL? (CHAIN) EOL? ;

value                       : literal | LPAR expression RPAR | variable ;
literal                     : matrixliteral | scalarliteral ;
matrixliteral               : LSQB rows+=rowliteral (BSLASH+ rows+=rowliteral)* RSQB ;
rowliteral                  : cols+=expression (COMMA cols+=expression)* ;
scalarliteral               : stringliteral | numliteral ;
stringliteral               : STRING ;
numliteral                  : INT | FLOAT ;

variable                    : cmnamespace varname (LSQB dimensionspec RSQB)? ;
cmnamespace                 : (file DOT)? (cmtypename DOT)? ;

args                        : expression (COMMA expression)* ;

cmtypedef                   : TYPE newtype attrblock EOL ;
attrblock                   : LBRA EOL? attrs+=attr+ RBRA ;
attr                        : attrname COLON typespec EOL ;

cmtypename                  : typeidentifier | primitive ;
varname                     : identifier ;
funcname                    : identifier ;
newtype                     : identifier ;
attrname                    : identifier ;

identifier                  : ID ;
typeidentifier              : TYPE_ID ;

primitive                   : T_INT | T_BOOL | T_STRING | T_FLOAT | functype ;
functype                    : LPAR argspecs+=typespec (COMMA argspecs+=typespec)* RPAR ARROW retspec=typespec ;
