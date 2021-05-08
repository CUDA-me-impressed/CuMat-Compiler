lexer grammar CuMatLexer;

fragment NEWLINE            : '\r\n' | '\n' | '\r' | '\u000B' | '\u000C' | '\u0085' | '\u2028' | '\u2029' ;
fragment ALPHA              : [a-zA-Z] ;
fragment POSDIGIT           : [1-9] ;
fragment DIGIT              : '0' | POSDIGIT ;
fragment ALPHANUM           : ALPHA | DIGIT ;
fragment ID_INITIAL         : ALPHA | '_' ;
fragment ID_TAIL            : ALPHA | DIGIT | '_' ;

EOL                         : NEWLINE+ ;
SPACE                       : (' ' | '\t' | '\u00A0' | '\u1680' | '\u2000' | '\u2001' | '\u2002' | '\u2003' | '\u2004'
                            | '\u2005' | '\u2006' | '\u2007' | '\u2008' | '\u2009' | '\u200A' | '\u202F' | '\u205F'
                            | '\u3000' | '\u180E' | '\u200B' | '\u2060' | '\uFEFF') -> skip ;

// Basic symbols
LPAR                        : '(';
RPAR                        : ')';
LSQB                        : '[';
RSQB                        : ']';
LBRA                        : '{';
RBRA                        : '}';
COMMA                       : ',';
DOT                         : '.';
COLON                       : ':';
BSLASH                      : '\\';
TYPE                        : 'type';
T_INT                       : 'int';
T_FLOAT                     : 'float';
T_BOOL                      : 'bool';
T_STRING                    : 'string';
FUNC                        : 'func';
RETURN                      : 'return' ;

IMPORT_OPEN                 : ('import' SPACE+) -> pushMode(IMPORT_MODE);

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
BNOT                        : '.!' ;
PLUS                        : '+' ;
MINUS                       : '-' ;
TIMES                       : '×' ;
STAR                        : '*' ;
DIV                         : '÷' | '/' ;
POW                         : '**' | '^' ;
MATM                        : '.*' ;
CHAIN                       : '|>' ;
ARROW                       : '->' ;
ASSIGN                      : '=' | '≔' ;

LAMBDA                      : '~' | 'λ' ;
IF                          : 'if' ;
THEN                        : 'then' ;
ELSE                        : 'else' ;

ID                          : ID_INITIAL (ID_TAIL*) ;
TYPE_ID                     : ID_INITIAL (ID_TAIL*) '\'' ;
INT                         : '0' | (POSDIGIT DIGIT*) ;
FLOAT                       : DIGIT+ ('.' DIGIT+)? ([eE][-+] DIGIT+)? ;
STRING                      : '"' (ALPHANUM | '-' | '.' | '/')* '"' ;
//STRING                      : '"' [^"\\]* ('\\'[\\|"]?[^"\\]+)*('\\'[\\|"]?)? '"' ;

mode IMPORT_MODE;
SLUG                        : (ALPHANUM | '-' | '.' )+ ;
DIRSEP                      : '/' ;
IMPORT_CLOSE                : (NEWLINE+) -> popMode;
