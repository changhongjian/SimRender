#include "Interface.h"

cls_c_param c_param;

EXT_C_DLL_VOID set_mp(char* nm, void * p) {
	string key(nm);
	c_param.mp[key] = p;
}

EXT_C_DLL void * get_mp(char* nm) {
	string key(nm);
	return c_param.mp[key];
}
