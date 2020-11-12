grammar CuMatGrammar;

fragment NEWLINE            : '\r\n' | '\n' | '\r' | '\u000B' | '\u000C' | '\u000D' | '\u0085' | '\u2028' | '\u2029' ;
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

SPACE                       : (' ' | '\t' | '\u00A0' | '\u1680' | '\u2000' | '\u2001' | '\u2002' | '\u2003' | '\u2004'
                            | '\u2005' | '\u2006' | '\u2007' | '\u2008' | '\u2009' | '\u200A' | '\u202F' | '\u205F'
                            | '\u3000' | '\u180E' | '\u200B' | '\u2060' | '\uFEFF') -> skip ;

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
directorylist               : (directories+=directory '/')* ;
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

block                       : '{' EOL? (assignments+=assignment EOL)* assignments+=assignment? EOL? '}' ;
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
op_bit                      : EOL? op=(BAND | BOR | BNOT) EOL? ;
op_sum                      : EOL? op=(PLUS | MINUS) EOL? ;
op_mult                     : EOL? op=(MUL | DIV) EOL? ;
op_pow                      : EOL? (POW) EOL? ;
op_mat                      : EOL? (MATM) EOL? ; // Dot product symbol??
op_neg                      : EOL? (MINUS) ;
op_not                      : EOL? (LNOT) ;
op_chain                    : EOL? (CHAIN) EOL? ;

value                       : literal | '(' expression ')' | variable ;
literal                     : matrixliteral | scalarliteral ;
matrixliteral               : '[' rows+=rowliteral ('\\'+ rows+=rowliteral)* ']' ;
rowliteral                  : cols+=expression (',' cols+=expression)* ;
scalarliteral               : stringliteral | numliteral ;
stringliteral               : STRING ;
numliteral                  : INT | FLOAT ;

variable                    : cmnamespace varname ('[' dimensionspec ']')? ;
cmnamespace                 : (file '.')? (cmtypename '.')? ;

args                        : expression (',' expression)* ;

cmtypedef                   : 'type' newtype attrblock EOL ;
attrblock                   : '{' EOL? attrs+=attr+ '}' ;
attr                        : attrname ':' typespec EOL ;

cmtypename                  : typeidentifier | primitive ;
varname                     : identifier ;
funcname                    : identifier ;
newtype                     : identifier ;
attrname                    : identifier ;

identifier                  : ID ;
typeidentifier              : TYPE_ID ;

primitive                   : 'int' | 'bool' | 'string' | 'float' | functype ;
functype                    : '(' argspecs+=typespec (',' argspecs+=typespec)* ')' ARROW retspec=typespec ;
