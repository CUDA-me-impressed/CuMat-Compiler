grammar CuMatGrammar;

fragment NEWLINE            : '\r\n' | '\n' | '\r' | 'U+000B' | 'U+000C' | 'U+000D' | 'U+0085' | 'U+2028' | 'U+2029' ;
fragment ALPHA              : [a-zA-Z] ;
fragment POSDIGIT           : [1-9] ;
fragment DIGIT              : '0' | POSDIGIT ;
fragment ALPHANUM           : ALPHA | DIGIT ;
fragment ID_INITIAL         : ALPHA | '_' ;
fragment ID_TAIL            : ALPHA | DIGIT | '_' ;

EOL                         : NEWLINE+ ;
SLUG                        : (ALPHANUM | '-' | '.' )+ ;
ID                          : ID_INITIAL (ID_TAIL*) ;
TYPE_ID                     : ID_INITIAL (ID_TAIL*) ('\''?) ;
INT                         : '0' | (POSDIGIT DIGIT*) ;
FLOAT                       : DIGIT+ ('.' DIGIT+)? ([eE][-+] DIGIT+)? ;
STRING                      : '"' [^"\\]* ('\\'[\\|"]?[^"\\]+)*('\\'[\\|"]?)? '"' ;

SPACE                       : (' ' | '\t' | 'U+00A0' | 'U+1680' | 'U+2000' | 'U+2001' | 'U+2002' | 'U+2003' | 'U+2004'
                            | 'U+2005' | 'U+2006' | 'U+2007' | 'U+2008' | 'U+2009' | 'U+200A' | 'U+202F' | 'U+205F'
                            | 'U+3000' | 'U+180E' | 'U+200B' | 'U+2060' | 'U+FEFF') -> skip ;

LOR                         : '||' | '⋁' | '∨';
LAND                        : '&&' | '⋀' | '∧';
LNOT                        : '!' | '¬' ;
EQ                          : '==' ;
NEQ                         : '!=' | '≠' ;
LT                          : '<' ;
GT                          : '>' ;
LTE                         : '<=' | '≤' ;
GTE                         : '>=' | '≥' ;
BOR                         : '|' ;
BAND                        : '&' ;
BNOT                        : '~' ;
PLUS                        : '+' ;
MINUS                       : '-' ;
MUL                         : '*' | '×' ;
DIV                         : '/' | '÷' ;
POW                         : '**' | '^' ;
MATM                        : '.*' ;
CHAIN                       : '|>' ;
ARROW                       : '->' ;
ASSIGN                      : '=' | '≔' ;

program                     : EOL* imports definitions EOF ;

imports                     : cmimport* ;
cmimport                    : 'import' path EOL ;
path                        : (directorylist '/')? file ;
directorylist               : (directory '/')* ;
directory                   : SLUG ;
file                        : SLUG ;

definitions                 : definition* ;
definition                  : (funcdef | cmtypedef | assignment) EOL ;

funcdef                     : 'func' signature EOL? block EOL ;
signature                   : typespec funcname ('(' arguments ')')? ;
arguments                   : (argument (',' argument)* )? ;
argument                    : typespec varname ;
typespec                    : cmtypename dimensionspec? ;
dimensionspec               : '[' dimension (',' dimension)* ']' ;
dimension                   : INT | '*' ;

block                       : '{' EOL? (assignment EOL)* assignment? EOL? '}' ;
assignment                  : varname ASSIGN expression ;

expression                  :
                            // Function Call
                              funcname '(' args ')'
                            // Chain operator
                            | <assoc=left> expression op=CHAIN expression
                            // Unary +/-
                            | op=(PLUS | MINUS) expression
                            // Unary logical/bitwise not
                            | op=(LNOT | BNOT) expression
                            // Matrix family
                            | <assoc=left> expression op=MATM expression
                            // Exponentiation family
                            | <assoc=left> expression op=POW expression
                            // Multiplication family
                            | <assoc=left> expression op=(MUL | DIV) expression
                            // Addition family
                            | <assoc=left> expression op=(PLUS | MINUS) expression
                            // Relational Operators
                            | <assoc=left> expression op=(LT | GT | LTE | GTE) expression
                            // Equational Operations
                            | <assoc=left> expression op=(EQ | NEQ) expression
                            // Bitwise Operators
                            | <assoc=left> expression op=BAND expression
                            | <assoc=left> expression op=BOR expression
                            // Logical Operators
                            | <assoc=left> expression op=LAND expression
                            | <assoc=left> expression op=LOR expression
                         // | lambda
                            | value;

value                       : literal | '(' expression ')' | variable ;
literal                     : matrixliteral | scalarliteral ;
matrixliteral               : '[' rowliteral ('\\'+ rowliteral)* ']' ;
rowliteral                  : expression (',' expression)* ;
scalarliteral               : stringliteral | numliteral ;
stringliteral               : STRING ;
numliteral                  : INT | FLOAT ;

variable                    : cmnamespace varname ('[' dimensionspec ']')? ;
cmnamespace                 : (file '.')? (cmtypename '.')? ;

args                        : expression (',' expression)* ;

cmtypedef                   : 'type' newtype attrblock EOL ;
attrblock                   : '{' EOL? attr+ '}' ;
attr                        : attrname ':' typespec EOL ;

cmtypename                  : typeidentifier | primitive ;
varname                     : identifier ;
funcname                    : identifier ;
newtype                     : identifier ;
attrname                    : identifier ;

identifier                  : ID ;
typeidentifier              : TYPE_ID ;

primitive                   : 'int' | 'bool' | 'string' | 'float' | functype ;
functype                    : '(' typespec (',' typespec)* ')' ARROW typespec ;
